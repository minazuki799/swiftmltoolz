from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="swiftmltoolz",
    version="0.1.0",
    author="Swift",
    description="A lightweight machine learning toolkit built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minazuki799/swiftmltoolz",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "scikit-learn>=1.0",
    ],
    python_requires=">=3.8",
    license="MIT",
)

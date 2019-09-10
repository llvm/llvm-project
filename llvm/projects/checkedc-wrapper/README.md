# Summary

Checked C is a research project to extend C with
static and dynamic checking to detect or prevent
common programming errors. 

This directory contains LLVM-specific files for 
testing the LLVM implementation of Checked C.
The files include build system files and LLVM
test infrastructure configuration files.

Checked C is a separate project from LLVM.
The specification for Checked C and the tests
for Checked C reside in a separate repo on
Github.   To use those tests, clone the
Checked C repo to this directory:

git clone https://github.com/Microsoft/checkedc

## Why this directory exists.

You might wonder why these files were not
placed directly into the Checked C repo,
allowing us to elide this directory entirely.
The build and test infrastructure configuration
files are derived directly from existing LLVM files,
so they are subject the LLVM license and copyright
terms.  The Checked C repo is subject to the MIT
conflicting licenses in the repo, just for a few files.



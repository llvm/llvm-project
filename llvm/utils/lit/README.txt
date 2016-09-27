===============================
 lit - A Software Testing Tool
===============================

lit is a portable tool for executing LLVM and Clang style test suites,
summarizing their results, and providing indication of failures. lit is designed
to be a lightweight testing tool with as simple a user interface as possible.

=====================
 Contributing to lit
=====================

Please read the TODO file in this directory for ideas on what to work on.
Before submitting patches, run the test suite to ensure nothing has regressed:

    # From within your LLVM source directory.
    utils/lit/lit.py \
        --path /path/to/your/llvm/build/bin \
        utils/lit/tests

Note that lit's tests depend on 'not' and 'FileCheck', LLVM utilities.
You will need to have built LLVM tools in order to run lit's test suite
successfully.


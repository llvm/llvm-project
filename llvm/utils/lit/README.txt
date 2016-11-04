===============================
 lit - A Software Testing Tool
===============================

lit is a portable tool for executing LLVM and Clang style test suites,
summarizing their results, and providing indication of failures. lit is designed
to be a lightweight testing tool with as simple a user interface as possible.

=====================
 Contributing to lit
=====================

Please browse the Test Suite > lit category in LLVM's Bugzilla for ideas on
what to work on.

Before submitting patches, run the test suite to ensure nothing has regressed:

    # From within your LLVM source directory.
    utils/lit/lit.py \
        --path /path/to/your/llvm/build/bin \
        utils/lit/tests

Note that lit's tests depend on 'not' and 'FileCheck', LLVM utilities.
You will need to have built LLVM tools in order to run lit's test suite
successfully.

You'll also want to confirm that lit continues to work when testing LLVM.
Follow the instructions in http://llvm.org/docs/TestingGuide.html to run the
regression test suite:

    make check-llvm

And be sure to run the llvm-lit wrapper script as well:

    /path/to/your/llvm/build/bin/llvm-lit utils/lit/tests

Finally, make sure lit works when installed via setuptools:

    python utils/lit/setup.py install
    lit --path /path/to/your/llvm/build/bin utils/lit/tests


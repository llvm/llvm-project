llvm-test-mustache-spec - LLVM tool to test Mustache Compliance Library
=========================================================================

llvm-test-mustache-spec tests the mustache spec conformance of the LLVM
mustache library. The spec can be found here: https://github.com/mustache/spec

To test against the spec, simply download the spec and pass the test JSON files
to the driver. Each spec file should have a list of tests for compliance with
the spec. These are loaded as test cases, and rendered with our Mustache
implementation, which is then compared against the expected output from the
spec.

The current implementation only supports non-optional parts of the spec, so
we do not expect any of the dynamic-names, inheritance, or lambda tests to
pass. Additionally, Triple Mustache is not supported. Unsupported tests are
marked as XFail and are removed from the XFail list as they are fixed.

    $ llvm-test-mustache-spec path/to/test/file.json path/to/test/file2.json ...

.. program:: llvm-test-mustache-spec

Outputs the number of test failures and successes in each of the test files.


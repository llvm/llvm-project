import os.path
import sys
import unittest


POSSIBLE_TEST_BINARIES = [
    'libreadline.so.5',
    'libreadline.so.6',
]

POSSIBLE_TEST_BINARY_PATHS = [
    '/usr/lib/debug',
    '/lib',
    '/usr/lib',
    '/usr/local/lib',
    '/lib/i386-linux-gnu',
]

class TestBase(unittest.TestCase):
    if sys.version_info.major == 2:
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    def get_test_binary(self):
        """Helper to obtain a test binary for object file testing.

        FIXME Support additional, highly-likely targets or create one
        ourselves.
        """
        for d in POSSIBLE_TEST_BINARY_PATHS:
            for lib in POSSIBLE_TEST_BINARIES:
                path = os.path.join(d, lib)

                if os.path.exists(path):
                    return path

        raise Exception('No suitable test binaries available!')
    get_test_binary.__test__ = False

    def get_test_file(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_file")

    def get_test_bc(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.bc")

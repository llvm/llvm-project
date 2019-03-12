#!/usr/bin/env python
import pickle
import unittest

from pexpect import ExceptionPexpect

class PickleTest(unittest.TestCase):
    def test_picking(self):
        e = ExceptionPexpect('Oh noes!')
        clone = pickle.loads(pickle.dumps(e))
        self.assertEqual(e.value, clone.value)

if __name__ == '__main__':
    unittest.main()

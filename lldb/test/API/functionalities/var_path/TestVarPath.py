"""
Make sure the getting a variable path works and doesn't crash.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestVarPath(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def verify_point(self, frame, var_name, var_typename, x_value, y_value):
        v = frame.GetValueForVariablePath(var_name)
        self.assertSuccess(v.GetError(), "Make sure we find '%s'" % (var_name))
        self.assertEqual(
            v.GetType().GetName(),
            var_typename,
            "Make sure '%s' has type '%s'" % (var_name, var_typename),
        )

        if "*" in var_typename:
            valid_prefix = var_name + "->"
            invalid_prefix = var_name + "."
        else:
            valid_prefix = var_name + "."
            invalid_prefix = var_name + "->"

        valid_x_path = valid_prefix + "x"
        valid_y_path = valid_prefix + "y"
        invalid_x_path = invalid_prefix + "x"
        invalid_y_path = invalid_prefix + "y"
        invalid_m_path = invalid_prefix + "m"

        v = frame.GetValueForVariablePath(valid_x_path)
        self.assertSuccess(v.GetError(), "Make sure we find '%s'" % (valid_x_path))
        self.assertEqual(
            v.GetValue(),
            str(x_value),
            "Make sure '%s' has a value of %i" % (valid_x_path, x_value),
        )
        self.assertEqual(
            v.GetType().GetName(),
            "int",
            "Make sure '%s' has type 'int'" % (valid_x_path),
        )
        v = frame.GetValueForVariablePath(invalid_x_path)
        self.assertTrue(
            v.GetError().Fail(), "Make sure we don't find '%s'" % (invalid_x_path)
        )

        v = frame.GetValueForVariablePath(valid_y_path)
        self.assertSuccess(v.GetError(), "Make sure we find '%s'" % (valid_y_path))
        self.assertEqual(
            v.GetValue(),
            str(y_value),
            "Make sure '%s' has a value of %i" % (valid_y_path, y_value),
        )
        self.assertEqual(
            v.GetType().GetName(),
            "int",
            "Make sure '%s' has type 'int'" % (valid_y_path),
        )
        v = frame.GetValueForVariablePath(invalid_y_path)
        self.assertTrue(
            v.GetError().Fail(), "Make sure we don't find '%s'" % (invalid_y_path)
        )

        v = frame.GetValueForVariablePath(invalid_m_path)
        self.assertTrue(
            v.GetError().Fail(), "Make sure we don't find '%s'" % (invalid_m_path)
        )

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.GetFrameAtIndex(0)
        v = frame.GetValueForVariablePath("no_such_variable")
        self.assertTrue(
            v.GetError().Fail(), "Make sure we don't find 'no_such_variable'"
        )

        # Test an instance
        self.verify_point(frame, "pt", "Point", 1, 2)
        # Test a pointer
        self.verify_point(frame, "pt_ptr", "Point *", 3030, 4040)
        # Test using a pointer as an array
        self.verify_point(frame, "pt_ptr[-1]", "Point", 1010, 2020)
        self.verify_point(frame, "pt_ptr[0]", "Point", 3030, 4040)
        self.verify_point(frame, "pt_ptr[1]", "Point", 5050, 6060)
        # Test arrays
        v = frame.GetValueForVariablePath("points")
        self.assertSuccess(v.GetError(), "Make sure we find 'points'")
        self.verify_point(frame, "points[0]", "Point", 1010, 2020)
        self.verify_point(frame, "points[1]", "Point", 3030, 4040)
        self.verify_point(frame, "points[2]", "Point", 5050, 6060)
        v = frame.GetValueForVariablePath("points[0]+5")
        self.assertTrue(
            v.GetError().Fail(),
            "Make sure we do not ignore characters between ']' and the end",
        )
        # Test a reference
        self.verify_point(frame, "pt_ref", "Point &", 1, 2)
        v = frame.GetValueForVariablePath("pt_sp")
        self.assertSuccess(v.GetError(), "Make sure we find 'pt_sp'")
        # Make sure we don't crash when looking for non existant child
        # in type with synthetic children. This used to cause a crash.
        if not self.isAArch64Windows():
            v = frame.GetValueForVariablePath("pt_sp->not_valid_child")
            self.assertTrue(
                v.GetError().Fail(), "Make sure we don't find 'pt_sp->not_valid_child'"
            )

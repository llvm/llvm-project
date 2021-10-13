"""Test the SBData APIs."""



from math import fabs
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBDataAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number('main.cpp', '// set breakpoint here')

    def test_byte_order_and_address_byte_size(self):
        """Test the SBData::SetData() to ensure the byte order and address
        byte size are obeyed"""
        addr_data = b'\x11\x22\x33\x44\x55\x66\x77\x88'
        error = lldb.SBError()
        data = lldb.SBData()
        data.SetData(error, addr_data, lldb.eByteOrderBig, 4)
        addr = data.GetAddress(error, 0)
        self.assertEqual(addr, 0x11223344);
        data.SetData(error, addr_data, lldb.eByteOrderBig, 8)
        addr = data.GetAddress(error, 0)
        self.assertEqual(addr, 0x1122334455667788);
        data.SetData(error, addr_data, lldb.eByteOrderLittle, 4)
        addr = data.GetAddress(error, 0)
        self.assertEqual(addr, 0x44332211);
        data.SetData(error, addr_data, lldb.eByteOrderLittle, 8)
        addr = data.GetAddress(error, 0)
        self.assertEqual(addr, 0x8877665544332211);

    def test_with_run_command(self):
        """Test the SBData APIs."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frame = thread.GetSelectedFrame()
        foobar = frame.FindVariable('foobar')
        self.assertTrue(foobar.IsValid())
        data = foobar.GetPointeeData(0, 2)
        offset = 0
        error = lldb.SBError()

        self.assert_data(data.GetUnsignedInt32, offset, 1)
        offset += 4
        low = data.GetSignedInt16(error, offset)
        self.assertTrue(error.Success())
        offset += 2
        high = data.GetSignedInt16(error, offset)
        self.assertTrue(error.Success())
        offset += 2
        self.assertTrue(
            (low == 9 and high == 0) or (
                low == 0 and high == 9),
            'foo[0].b == 9')
        self.assertTrue(
            fabs(
                data.GetFloat(
                    error,
                    offset) -
                3.14) < 1,
            'foo[0].c == 3.14')
        self.assertTrue(error.Success())
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 8)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 5)
        offset += 4

        self.runCmd("n")

        offset = 16

        self.assert_data(data.GetUnsignedInt32, offset, 5)

        data = foobar.GetPointeeData(1, 1)

        offset = 0

        self.assert_data(data.GetSignedInt32, offset, 8)
        offset += 4
        self.assert_data(data.GetSignedInt32, offset, 7)
        offset += 8
        self.assertTrue(
            data.GetUnsignedInt32(
                error,
                offset) == 0,
            'do not read beyond end')
        self.assertTrue(not error.Success())
        error.Clear()  # clear the error for the next test

        star_foobar = foobar.Dereference()
        self.assertTrue(star_foobar.IsValid())

        data = star_foobar.GetData()

        offset = 0
        self.assert_data(data.GetUnsignedInt32, offset, 1)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 9)

        foobar_addr = star_foobar.GetLoadAddress()
        foobar_addr += 12

        # http://llvm.org/bugs/show_bug.cgi?id=11579
        # lldb::SBValue::CreateValueFromAddress does not verify SBType::GetPointerType succeeds
        # This should not crash LLDB.
        nothing = foobar.CreateValueFromAddress(
            "nothing", foobar_addr, star_foobar.GetType().GetBasicType(
                lldb.eBasicTypeInvalid))

        new_foobar = foobar.CreateValueFromAddress(
            "f00", foobar_addr, star_foobar.GetType())
        self.assertTrue(new_foobar.IsValid())
        data = new_foobar.GetData()

        self.assertEqual(data.uint32[0], 8, 'then foo[1].a == 8')
        self.assertEqual(data.uint32[1], 7, 'then foo[1].b == 7')
        # exploiting that sizeof(uint32) == sizeof(float)
        self.assertTrue(fabs(data.float[2] - 3.14) < 1, 'foo[1].c == 3.14')

        self.runCmd("n")

        offset = 0
        self.assert_data(data.GetUnsignedInt32, offset, 8)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 7)
        offset += 4
        self.assertTrue(
            fabs(
                data.GetFloat(
                    error,
                    offset) -
                3.14) < 1,
            'foo[1].c == 3.14')
        self.assertTrue(error.Success())

        data = new_foobar.GetData()

        offset = 0
        self.assert_data(data.GetUnsignedInt32, offset, 8)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 7)
        offset += 4
        self.assertTrue(
            fabs(
                data.GetFloat(
                    error,
                    offset) -
                6.28) < 1,
            'foo[1].c == 6.28')
        self.assertTrue(error.Success())

        self.runCmd("n")

        barfoo = frame.FindVariable('barfoo')

        data = barfoo.GetData()
        offset = 0
        self.assert_data(data.GetUnsignedInt32, offset, 1)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 2)
        offset += 4
        self.assertTrue(
            fabs(
                data.GetFloat(
                    error,
                    offset) -
                3) < 1,
            'barfoo[0].c == 3')
        self.assertTrue(error.Success())
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 4)
        offset += 4
        self.assert_data(data.GetUnsignedInt32, offset, 5)
        offset += 4
        self.assertTrue(
            fabs(
                data.GetFloat(
                    error,
                    offset) -
                6) < 1,
            'barfoo[1].c == 6')
        self.assertTrue(error.Success())

        new_object = barfoo.CreateValueFromData(
            "new_object", data, barfoo.GetType().GetBasicType(
                lldb.eBasicTypeInt))
        self.assertEqual(new_object.GetValue(), "1", 'new_object == 1')

        if data.GetByteOrder() == lldb.eByteOrderBig:
            data.SetData(
                error,
                '\0\0\0A',
                data.GetByteOrder(),
                data.GetAddressByteSize())
        else:
            data.SetData(
                error,
                'A\0\0\0',
                data.GetByteOrder(),
                data.GetAddressByteSize())
        self.assertTrue(error.Success())

        data2 = lldb.SBData()
        data2.SetData(
            error,
            'BCD',
            data.GetByteOrder(),
            data.GetAddressByteSize())
        self.assertTrue(error.Success())

        data.Append(data2)

        # this breaks on EBCDIC
        offset = 0
        self.assert_data(data.GetUnsignedInt32, offset, 65)
        offset += 4
        self.assert_data(data.GetUnsignedInt8, offset, 66)
        offset += 1
        self.assert_data(data.GetUnsignedInt8, offset, 67)
        offset += 1
        self.assert_data(data.GetUnsignedInt8, offset, 68)
        offset += 1

        # check the new API calls introduced per LLVM llvm.org/prenhancement request
        # 11619 (Allow creating SBData values from arrays or primitives in
        # Python)

        hello_str = "hello!"
        data2 = lldb.SBData.CreateDataFromCString(
            process.GetByteOrder(), process.GetAddressByteSize(), hello_str)
        self.assertEqual(len(data2.uint8), len(hello_str))
        self.assertEqual(data2.uint8[0], 104, 'h == 104')
        self.assertEqual(data2.uint8[1], 101, 'e == 101')
        self.assertEqual(data2.uint8[2], 108, 'l == 108')
        self.assert_data(data2.GetUnsignedInt8, 3, 108)  # l
        self.assertEqual(data2.uint8[4], 111, 'o == 111')
        self.assert_data(data2.GetUnsignedInt8, 5, 33)  # !

        uint_lists = [[1, 2, 3, 4, 5], [int(i) for i in [1, 2, 3, 4, 5]]]
        int_lists = [[2, -2], [int(i) for i in [2, -2]]]

        for l in uint_lists:
            data2 = lldb.SBData.CreateDataFromUInt64Array(
                process.GetByteOrder(), process.GetAddressByteSize(), l)
            self.assert_data(data2.GetUnsignedInt64, 0, 1)
            self.assert_data(data2.GetUnsignedInt64, 8, 2)
            self.assert_data(data2.GetUnsignedInt64, 16, 3)
            self.assert_data(data2.GetUnsignedInt64, 24, 4)
            self.assert_data(data2.GetUnsignedInt64, 32, 5)

            self.assertTrue(
                data2.uint64s == [
                    1,
                    2,
                    3,
                    4,
                    5],
                'read_data_helper failure: data2 == [1,2,3,4,5]')

        for l in int_lists:
            data2 = lldb.SBData.CreateDataFromSInt32Array(
                process.GetByteOrder(), process.GetAddressByteSize(), l)
            self.assertTrue(
                data2.sint32[
                    0:2] == [
                    2, -2], 'signed32 data2 = [2,-2]')

        data2.Append(
            lldb.SBData.CreateDataFromSInt64Array(
                process.GetByteOrder(),
                process.GetAddressByteSize(),
                int_lists[0]))
        self.assert_data(data2.GetSignedInt32, 0, 2)
        self.assert_data(data2.GetSignedInt32, 4, -2)
        self.assertTrue(
            data2.sint64[
                1:3] == [
                2, -2], 'signed64 data2 = [2,-2]')

        for l in int_lists:
            data2 = lldb.SBData.CreateDataFromSInt64Array(
                process.GetByteOrder(), process.GetAddressByteSize(), l)
            self.assert_data(data2.GetSignedInt64, 0, 2)
            self.assert_data(data2.GetSignedInt64, 8, -2)
            self.assertTrue(
                data2.sint64[
                    0:2] == [
                    2, -2], 'signed64 data2 = [2,-2]')

        for l in uint_lists:
            data2 = lldb.SBData.CreateDataFromUInt32Array(
                process.GetByteOrder(), process.GetAddressByteSize(), l)
            self.assert_data(data2.GetUnsignedInt32, 0, 1)
            self.assert_data(data2.GetUnsignedInt32, 4, 2)
            self.assert_data(data2.GetUnsignedInt32, 8, 3)
            self.assert_data(data2.GetUnsignedInt32, 12, 4)
            self.assert_data(data2.GetUnsignedInt32, 16, 5)

        bool_list = [True, True, False, False, True, False]

        data2 = lldb.SBData.CreateDataFromSInt32Array(
            process.GetByteOrder(), process.GetAddressByteSize(), bool_list)
        self.assertTrue(
            data2.sint32[
                0:6] == [
                1,
                1,
                0,
                0,
                1,
                0],
            'signed32 data2 = [1, 1, 0, 0, 1, 0]')

        data2 = lldb.SBData.CreateDataFromUInt32Array(
            process.GetByteOrder(), process.GetAddressByteSize(), bool_list)
        self.assertTrue(
            data2.uint32[
                0:6] == [
                1,
                1,
                0,
                0,
                1,
                0],
            'unsigned32 data2 = [1, 1, 0, 0, 1, 0]')

        data2 = lldb.SBData.CreateDataFromSInt64Array(
            process.GetByteOrder(), process.GetAddressByteSize(), bool_list)
        self.assertTrue(
            data2.sint64[
                0:6] == [
                1,
                1,
                0,
                0,
                1,
                0],
            'signed64 data2 = [1, 1, 0, 0, 1, 0]')

        data2 = lldb.SBData.CreateDataFromUInt64Array(
            process.GetByteOrder(), process.GetAddressByteSize(), bool_list)
        self.assertTrue(
            data2.uint64[
                0:6] == [
                1,
                1,
                0,
                0,
                1,
                0],
            'signed64 data2 = [1, 1, 0, 0, 1, 0]')

        data2 = lldb.SBData.CreateDataFromDoubleArray(
            process.GetByteOrder(), process.GetAddressByteSize(), [
                3.14, 6.28, 2.71])
        self.assertTrue(
            fabs(
                data2.GetDouble(
                    error,
                    0) -
                3.14) < 0.5,
            'double data2[0] = 3.14')
        self.assertTrue(error.Success())
        self.assertTrue(
            fabs(
                data2.GetDouble(
                    error,
                    8) -
                6.28) < 0.5,
            'double data2[1] = 6.28')
        self.assertTrue(error.Success())
        self.assertTrue(
            fabs(
                data2.GetDouble(
                    error,
                    16) -
                2.71) < 0.5,
            'double data2[2] = 2.71')
        self.assertTrue(error.Success())

        data2 = lldb.SBData()

        data2.SetDataFromCString(hello_str)
        self.assertEqual(len(data2.uint8), len(hello_str))
        self.assert_data(data2.GetUnsignedInt8, 0, 104)
        self.assert_data(data2.GetUnsignedInt8, 1, 101)
        self.assert_data(data2.GetUnsignedInt8, 2, 108)
        self.assert_data(data2.GetUnsignedInt8, 3, 108)
        self.assert_data(data2.GetUnsignedInt8, 4, 111)
        self.assert_data(data2.GetUnsignedInt8, 5, 33)

        data2.SetDataFromUInt64Array([1, 2, 3, 4, 5])
        self.assert_data(data2.GetUnsignedInt64, 0, 1)
        self.assert_data(data2.GetUnsignedInt64, 8, 2)
        self.assert_data(data2.GetUnsignedInt64, 16, 3)
        self.assert_data(data2.GetUnsignedInt64, 24, 4)
        self.assert_data(data2.GetUnsignedInt64, 32, 5)

        self.assertEqual(
            data2.uint64[0], 1,
            'read_data_helper failure: set data2[0] = 1')
        self.assertEqual(
            data2.uint64[1], 2,
            'read_data_helper failure: set data2[1] = 2')
        self.assertEqual(
            data2.uint64[2], 3,
            'read_data_helper failure: set data2[2] = 3')
        self.assertEqual(
            data2.uint64[3], 4,
            'read_data_helper failure: set data2[3] = 4')
        self.assertEqual(
            data2.uint64[4], 5,
            'read_data_helper failure: set data2[4] = 5')

        self.assertTrue(
            data2.uint64[
                0:2] == [
                1,
                2],
            'read_data_helper failure: set data2[0:2] = [1,2]')

        data2.SetDataFromSInt32Array([2, -2])
        self.assert_data(data2.GetSignedInt32, 0, 2)
        self.assert_data(data2.GetSignedInt32, 4, -2)

        data2.SetDataFromSInt64Array([2, -2])
        self.assert_data(data2.GetSignedInt64, 0, 2)
        self.assert_data(data2.GetSignedInt64, 8, -2)

        data2.SetDataFromUInt32Array([1, 2, 3, 4, 5])
        self.assert_data(data2.GetUnsignedInt32, 0, 1)
        self.assert_data(data2.GetUnsignedInt32, 4, 2)
        self.assert_data(data2.GetUnsignedInt32, 8, 3)
        self.assert_data(data2.GetUnsignedInt32, 12, 4)
        self.assert_data(data2.GetUnsignedInt32, 16, 5)

        self.assertEqual(
            data2.uint32[0], 1,
            'read_data_helper failure: set 32-bit data2[0] = 1')
        self.assertEqual(
            data2.uint32[1], 2,
            'read_data_helper failure: set 32-bit data2[1] = 2')
        self.assertEqual(
            data2.uint32[2], 3,
            'read_data_helper failure: set 32-bit data2[2] = 3')
        self.assertEqual(
            data2.uint32[3], 4,
            'read_data_helper failure: set 32-bit data2[3] = 4')
        self.assertEqual(
            data2.uint32[4], 5,
            'read_data_helper failure: set 32-bit data2[4] = 5')

        data2.SetDataFromDoubleArray([3.14, 6.28, 2.71])
        self.assertTrue(fabs(data2.GetDouble(error, 0) - 3.14)
                        < 0.5, 'set double data2[0] = 3.14')
        self.assertTrue(fabs(data2.GetDouble(error, 8) - 6.28)
                        < 0.5, 'set double data2[1] = 6.28')
        self.assertTrue(fabs(data2.GetDouble(error, 16) - 2.71)
                        < 0.5, 'set double data2[2] = 2.71')

        self.assertTrue(
            fabs(
                data2.double[0] -
                3.14) < 0.5,
            'read_data_helper failure: set double data2[0] = 3.14')
        self.assertTrue(
            fabs(
                data2.double[1] -
                6.28) < 0.5,
            'read_data_helper failure: set double data2[1] = 6.28')
        self.assertTrue(
            fabs(
                data2.double[2] -
                2.71) < 0.5,
            'read_data_helper failure: set double data2[2] = 2.71')

    def assert_data(self, func, arg, expected):
        """ Asserts func(SBError error, arg) == expected. """
        error = lldb.SBError()
        result = func(error, arg)
        if not error.Success():
            stream = lldb.SBStream()
            error.GetDescription(stream)
            self.assertTrue(
                error.Success(), "%s(error, %s) did not succeed: %s" %
                (func.__name__, arg, stream.GetData()))
        self.assertTrue(
            expected == result, "%s(error, %s) == %s != %s" %
            (func.__name__, arg, result, expected))

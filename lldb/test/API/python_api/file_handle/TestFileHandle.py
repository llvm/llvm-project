"""
Test lldb Python API for file handles.
"""


import os
import io
import re
import sys
from contextlib import contextmanager

import lldb
from lldbsuite.test import  lldbtest
from lldbsuite.test.decorators import *

class OhNoe(Exception):
    pass

class BadIO(io.TextIOBase):
    @property
    def closed(self):
        return False
    def writable(self):
        return True
    def readable(self):
        return True
    def write(self, s):
        raise OhNoe('OH NOE')
    def read(self, n):
        raise OhNoe("OH NOE")
    def flush(self):
        raise OhNoe('OH NOE')

# This class will raise an exception while it's being
# converted into a C++ object by swig
class ReallyBadIO(io.TextIOBase):
    def fileno(self):
        return 999
    def writable(self):
        raise OhNoe("OH NOE!!!")

class MutableBool():
    def __init__(self, value):
        self.value = value
    def set(self, value):
        self.value = bool(value)
    def __bool__(self):
        return self.value

class FlushTestIO(io.StringIO):
    def __init__(self, mutable_flushed, mutable_closed):
        super(FlushTestIO, self).__init__()
        self.mut_flushed = mutable_flushed
        self.mut_closed = mutable_closed
    def close(self):
        self.mut_closed.set(True)
        return super(FlushTestIO, self).close()
    def flush(self):
        self.mut_flushed.set(True)
        return super(FlushTestIO, self).flush()

@contextmanager
def replace_stdout(new):
    old = sys.stdout
    sys.stdout = new
    try:
        yield
    finally:
        sys.stdout = old

def readStrippedLines(f):
    def i():
        for line in f:
            line = line.strip()
            if line:
                yield line
    return list(i())


class FileHandleTestCase(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = lldbtest.Base.compute_mydir(__file__)

    # The way normal tests evaluate debugger commands is
    # by using a SBCommandInterpreter directly, which captures
    # the output in a result object.   For many of tests tests
    # we want the debugger to write the  output directly to
    # its I/O streams like it would have done interactively.
    #
    # For this reason we also define handleCmd() here, even though
    # it is similar to runCmd().

    def setUp(self):
        super(FileHandleTestCase, self).setUp()
        self.out_filename = self.getBuildArtifact('output')
        self.in_filename = self.getBuildArtifact('input')

    def tearDown(self):
        super(FileHandleTestCase, self).tearDown()
        for name in (self.out_filename, self.in_filename):
            if os.path.exists(name):
                os.unlink(name)

    # Similar to runCmd(), but letting the debugger just print the results
    # instead of collecting them.
    def handleCmd(self, cmd, check=True, collect_result=True):
        assert not check or collect_result
        ret = lldb.SBCommandReturnObject()
        if collect_result:
            interpreter = self.dbg.GetCommandInterpreter()
            interpreter.HandleCommand(cmd, ret)
        else:
            self.dbg.HandleCommand(cmd)
        self.dbg.GetOutputFile().Flush()
        self.dbg.GetErrorFile().Flush()
        if collect_result and check:
            self.assertTrue(ret.Succeeded())
        return ret.GetOutput()


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_legacy_file_out_script(self):
        with open(self.out_filename, 'w') as f:
            self.dbg.SetOutputFileHandle(f, False)
            # scripts print to output even if you capture the results
            # I'm not sure I love that behavior, but that's the way
            # it's been for a long time.  That's why this test works
            # even with collect_result=True.
            self.handleCmd('script 1+1')
            self.dbg.GetOutputFileHandle().write('FOO\n')
            self.dbg.GetOutputFileHandle().flush()
        with open(self.out_filename, 'r') as f:
            self.assertEqual(readStrippedLines(f), ['2', 'FOO'])


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_legacy_file_out(self):
        with open(self.out_filename, 'w') as f:
            self.dbg.SetOutputFileHandle(f, False)
            self.handleCmd('p/x 3735928559', collect_result=False, check=False)
        with open(self.out_filename, 'r') as f:
            self.assertIn('deadbeef', f.read())

    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_legacy_file_err_with_get(self):
        with open(self.out_filename, 'w') as f:
            self.dbg.SetErrorFileHandle(f, False)
            self.handleCmd('lolwut', check=False, collect_result=False)
            f2 = self.dbg.GetErrorFileHandle()
            f2.write('FOOBAR\n')
            f2.flush()
        with open(self.out_filename, 'r') as f:
            errors = f.read()
            self.assertTrue(re.search(r'error:.*lolwut', errors))
            self.assertTrue(re.search(r'FOOBAR', errors))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_legacy_file_err(self):
        with open(self.out_filename, 'w') as f:
            self.dbg.SetErrorFileHandle(f, False)
            self.handleCmd('lol', check=False, collect_result=False)
        with open(self.out_filename, 'r') as f:
            self.assertIn("is not a valid command", f.read())


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_legacy_file_error(self):
        with open(self.out_filename, 'w') as f:
            self.dbg.SetErrorFileHandle(f, False)
            self.handleCmd('lolwut', check=False, collect_result=False)
        with open(self.out_filename, 'r') as f:
            errors = f.read()
            self.assertTrue(re.search(r'error:.*lolwut', errors))

    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_type_errors(self):
        sbf = lldb.SBFile()
        self.assertRaises(Exception, sbf.Write, None)
        self.assertRaises(Exception, sbf.Read, None)
        self.assertRaises(Exception, sbf.Read, b'this bytes is not mutable')
        self.assertRaises(Exception, sbf.Write, u"ham sandwich")
        self.assertRaises(Exception, sbf.Read, u"ham sandwich")


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_fileno(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f.fileno(), "w", False)
            self.assertTrue(sbf.IsValid())
            e, n = sbf.Write(b'FOO\nBAR')
            self.assertTrue(e.Success())
            self.assertEqual(n, 7)
            sbf.Close()
            self.assertFalse(sbf.IsValid())
        with open(self.out_filename, 'r') as f:
            self.assertEqual(readStrippedLines(f), ['FOO', 'BAR'])


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f)
            e, n = sbf.Write(b'FOO\n')
            self.assertTrue(e.Success())
            self.assertEqual(n, 4)
            sbf.Close()
            self.assertTrue(f.closed)
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), 'FOO')


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_read_fileno(self):
        with open(self.out_filename, 'w') as f:
            f.write('FOO')
        with open(self.out_filename, 'r') as f:
            sbf = lldb.SBFile(f.fileno(), "r", False)
            self.assertTrue(sbf.IsValid())
            buffer = bytearray(100)
            e, n = sbf.Read(buffer)
            self.assertTrue(e.Success())
            self.assertEqual(buffer[:n], b'FOO')


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_read(self):
        with open(self.out_filename, 'w') as f:
            f.write('foo')
        with open(self.out_filename, 'r') as f:
            sbf = lldb.SBFile(f)
            buf = bytearray(100)
            e, n = sbf.Read(buf)
            self.assertTrue(e.Success())
            self.assertEqual(n, 3)
            self.assertEqual(buf[:n], b'foo')
            sbf.Close()
            self.assertTrue(f.closed)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_fileno_out(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f.fileno(), "w", False)
            status = self.dbg.SetOutputFile(sbf)
            self.assertTrue(status.Success())
            self.handleCmd('script 1+2')
            self.dbg.GetOutputFile().Write(b'quux')
            self.dbg.GetOutputFile().Flush()

        with open(self.out_filename, 'r') as f:
            self.assertEqual(readStrippedLines(f), ['3', 'quux'])


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_fileno_help(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f.fileno(), "w", False)
            status = self.dbg.SetOutputFile(sbf)
            self.assertTrue(status.Success())
            self.handleCmd("help help", collect_result=False, check=False)
        with open(self.out_filename, 'r') as f:
            self.assertTrue(re.search(r'Show a list of all debugger commands', f.read()))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_help(self):
        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetOutputFile(lldb.SBFile(f))
            self.assertTrue(status.Success())
            self.handleCmd("help help", check=False, collect_result=False)
        with open(self.out_filename, 'r') as f:
            self.assertIn('Show a list of all debugger commands', f.read())


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_immediate(self):
        with open(self.out_filename, 'w') as f:
            ret = lldb.SBCommandReturnObject()
            ret.SetImmediateOutputFile(f)
            interpreter = self.dbg.GetCommandInterpreter()
            interpreter.HandleCommand("help help", ret)
            # make sure the file wasn't closed early.
            f.write("\nQUUX\n")
        ret = None # call destructor and flush streams
        with open(self.out_filename, 'r') as f:
            output = f.read()
            self.assertTrue(re.search(r'Show a list of all debugger commands', output))
            self.assertTrue(re.search(r'QUUX', output))


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_immediate_string(self):
        f = io.StringIO()
        ret = lldb.SBCommandReturnObject()
        ret.SetImmediateOutputFile(f)
        interpreter = self.dbg.GetCommandInterpreter()
        interpreter.HandleCommand("help help", ret)
        # make sure the file wasn't closed early.
        f.write("\nQUUX\n")
        ret = None # call destructor and flush streams
        output = f.getvalue()
        self.assertTrue(re.search(r'Show a list of all debugger commands', output))
        self.assertTrue(re.search(r'QUUX', output))


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_immediate_sbfile_string(self):
        f = io.StringIO()
        ret = lldb.SBCommandReturnObject()
        ret.SetImmediateOutputFile(lldb.SBFile(f))
        interpreter = self.dbg.GetCommandInterpreter()
        interpreter.HandleCommand("help help", ret)
        output = f.getvalue()
        ret = None # call destructor and flush streams
        # sbfile default constructor doesn't borrow the file
        self.assertTrue(f.closed)
        self.assertTrue(re.search(r'Show a list of all debugger commands', output))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_fileno_inout(self):
        with open(self.in_filename, 'w') as f:
            f.write("help help\n")

        with open(self.out_filename, 'w') as outf, open(self.in_filename, 'r') as inf:

            outsbf = lldb.SBFile(outf.fileno(), "w", False)
            status = self.dbg.SetOutputFile(outsbf)
            self.assertTrue(status.Success())

            insbf = lldb.SBFile(inf.fileno(), "r", False)
            status = self.dbg.SetInputFile(insbf)
            self.assertTrue(status.Success())

            opts = lldb.SBCommandInterpreterRunOptions()
            self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)
            self.dbg.GetOutputFile().Flush()

        with open(self.out_filename, 'r') as f:
            self.assertTrue(re.search(r'Show a list of all debugger commands', f.read()))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_inout(self):
        with open(self.in_filename, 'w') as f:
            f.write("help help\n")
        with  open(self.out_filename, 'w') as outf, \
              open(self.in_filename, 'r') as inf:
            status = self.dbg.SetOutputFile(lldb.SBFile(outf))
            self.assertTrue(status.Success())
            status = self.dbg.SetInputFile(lldb.SBFile(inf))
            self.assertTrue(status.Success())
            opts = lldb.SBCommandInterpreterRunOptions()
            self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)
            self.dbg.GetOutputFile().Flush()
        with open(self.out_filename, 'r') as f:
            output = f.read()
            self.assertIn('Show a list of all debugger commands', output)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_binary_inout(self):
        with open(self.in_filename, 'w') as f:
            f.write("help help\n")
        with  open(self.out_filename, 'wb') as outf, \
              open(self.in_filename, 'rb') as inf:
            status = self.dbg.SetOutputFile(lldb.SBFile(outf))
            self.assertTrue(status.Success())
            status = self.dbg.SetInputFile(lldb.SBFile(inf))
            self.assertTrue(status.Success())
            opts = lldb.SBCommandInterpreterRunOptions()
            self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)
            self.dbg.GetOutputFile().Flush()
        with open(self.out_filename, 'r') as f:
            output = f.read()
            self.assertIn('Show a list of all debugger commands', output)


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_string_inout(self):
        inf = io.StringIO("help help\np/x ~0\n")
        outf = io.StringIO()
        status = self.dbg.SetOutputFile(lldb.SBFile(outf))
        self.assertTrue(status.Success())
        status = self.dbg.SetInputFile(lldb.SBFile(inf))
        self.assertTrue(status.Success())
        opts = lldb.SBCommandInterpreterRunOptions()
        self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)
        self.dbg.GetOutputFile().Flush()
        output = outf.getvalue()
        self.assertIn('Show a list of all debugger commands', output)
        self.assertIn('0xfff', output)


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_bytes_inout(self):
        inf = io.BytesIO(b"help help\nhelp b\n")
        outf = io.BytesIO()
        status = self.dbg.SetOutputFile(lldb.SBFile(outf))
        self.assertTrue(status.Success())
        status = self.dbg.SetInputFile(lldb.SBFile(inf))
        self.assertTrue(status.Success())
        opts = lldb.SBCommandInterpreterRunOptions()
        self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)
        self.dbg.GetOutputFile().Flush()
        output = outf.getvalue()
        self.assertIn(b'Show a list of all debugger commands', output)
        self.assertIn(b'Set a breakpoint', output)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_fileno_error(self):
        with open(self.out_filename, 'w') as f:

            sbf = lldb.SBFile(f.fileno(), 'w', False)
            status = self.dbg.SetErrorFile(sbf)
            self.assertTrue(status.Success())

            self.handleCmd('lolwut', check=False, collect_result=False)

            self.dbg.GetErrorFile().Write(b'\nzork\n')

        with open(self.out_filename, 'r') as f:
            errors = f.read()
            self.assertTrue(re.search(r'error:.*lolwut', errors))
            self.assertTrue(re.search(r'zork', errors))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_replace_stdout(self):
        f = io.StringIO()
        with replace_stdout(f):
            self.assertEqual(sys.stdout, f)
            self.handleCmd('script sys.stdout.write("lol")',
                collect_result=False, check=False)
            self.assertEqual(sys.stdout, f)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_replace_stdout_with_nonfile(self):
        f = io.StringIO()
        with replace_stdout(f):
            class Nothing():
                pass
            with replace_stdout(Nothing):
                self.assertEqual(sys.stdout, Nothing)
                self.handleCmd('script sys.stdout.write("lol")',
                    check=False, collect_result=False)
                self.assertEqual(sys.stdout, Nothing)
            sys.stdout.write(u"FOO")
        self.assertEqual(f.getvalue(), "FOO")


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_borrowed(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile.Create(f, borrow=True)
            e, n = sbf.Write(b'FOO')
            self.assertTrue(e.Success())
            self.assertEqual(n, 3)
            sbf.Close()
            self.assertFalse(f.closed)
            f.write('BAR\n')
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), 'FOOBAR')



    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_forced(self):
        with open(self.out_filename, 'w') as f:
            written = MutableBool(False)
            orig_write = f.write
            def mywrite(x):
                written.set(True)
                return orig_write(x)
            f.write = mywrite
            sbf = lldb.SBFile.Create(f, force_io_methods=True)
            e, n = sbf.Write(b'FOO')
            self.assertTrue(written)
            self.assertTrue(e.Success())
            self.assertEqual(n, 3)
            sbf.Close()
            self.assertTrue(f.closed)
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), 'FOO')


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_forced_borrowed(self):
        with open(self.out_filename, 'w') as f:
            written = MutableBool(False)
            orig_write = f.write
            def mywrite(x):
                written.set(True)
                return orig_write(x)
            f.write = mywrite
            sbf = lldb.SBFile.Create(f, borrow=True, force_io_methods=True)
            e, n = sbf.Write(b'FOO')
            self.assertTrue(written)
            self.assertTrue(e.Success())
            self.assertEqual(n, 3)
            sbf.Close()
            self.assertFalse(f.closed)
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), 'FOO')


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_string(self):
        f = io.StringIO()
        sbf = lldb.SBFile(f)
        e, n = sbf.Write(b'FOO')
        self.assertEqual(f.getvalue().strip(), "FOO")
        self.assertTrue(e.Success())
        self.assertEqual(n, 3)
        sbf.Close()
        self.assertTrue(f.closed)


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_string_out(self):
        f = io.StringIO()
        status = self.dbg.SetOutputFile(f)
        self.assertTrue(status.Success())
        self.handleCmd("script 'foobar'")
        self.assertEqual(f.getvalue().strip(), "'foobar'")


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_string_error(self):
        f = io.StringIO()
        status = self.dbg.SetErrorFile(f)
        self.assertTrue(status.Success())
        self.handleCmd('lolwut', check=False, collect_result=False)
        errors = f.getvalue()
        self.assertTrue(re.search(r'error:.*lolwut', errors))


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_write_bytes(self):
        f = io.BytesIO()
        sbf = lldb.SBFile(f)
        e, n = sbf.Write(b'FOO')
        self.assertEqual(f.getvalue().strip(), b"FOO")
        self.assertTrue(e.Success())
        self.assertEqual(n, 3)
        sbf.Close()
        self.assertTrue(f.closed)

    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_read_string(self):
        f = io.StringIO('zork')
        sbf = lldb.SBFile(f)
        buf = bytearray(100)
        e, n = sbf.Read(buf)
        self.assertTrue(e.Success())
        self.assertEqual(buf[:n], b'zork')


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_read_string_one_byte(self):
        f = io.StringIO('z')
        sbf = lldb.SBFile(f)
        buf = bytearray(1)
        e, n = sbf.Read(buf)
        self.assertTrue(e.Fail())
        self.assertEqual(n, 0)
        self.assertEqual(e.GetCString(), "can't read less than 6 bytes from a utf8 text stream")


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_read_bytes(self):
        f = io.BytesIO(b'zork')
        sbf = lldb.SBFile(f)
        buf = bytearray(100)
        e, n = sbf.Read(buf)
        self.assertTrue(e.Success())
        self.assertEqual(buf[:n], b'zork')


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_out(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f)
            status = self.dbg.SetOutputFile(sbf)
            self.assertTrue(status.Success())
            self.handleCmd('script 2+2')
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), '4')


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_file_out(self):
        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetOutputFile(f)
            self.assertTrue(status.Success())
            self.handleCmd('script 2+2')
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), '4')


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbfile_error(self):
        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f)
            status = self.dbg.SetErrorFile(sbf)
            self.assertTrue(status.Success())
            self.handleCmd('lolwut', check=False, collect_result=False)
        with open(self.out_filename, 'r') as f:
            errors = f.read()
            self.assertTrue(re.search(r'error:.*lolwut', errors))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_file_error(self):
        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetErrorFile(f)
            self.assertTrue(status.Success())
            self.handleCmd('lolwut', check=False, collect_result=False)
        with open(self.out_filename, 'r') as f:
            errors = f.read()
            self.assertTrue(re.search(r'error:.*lolwut', errors))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_exceptions(self):
        self.assertRaises(Exception, lldb.SBFile, None)
        self.assertRaises(Exception, lldb.SBFile, "ham sandwich")
        if sys.version_info[0] < 3:
            self.assertRaises(Exception, lldb.SBFile, ReallyBadIO())
        else:
            self.assertRaises(OhNoe, lldb.SBFile, ReallyBadIO())
            error, n = lldb.SBFile(BadIO()).Write(b"FOO")
            self.assertEqual(n, 0)
            self.assertTrue(error.Fail())
            self.assertIn('OH NOE', error.GetCString())
            error, n = lldb.SBFile(BadIO()).Read(bytearray(100))
            self.assertEqual(n, 0)
            self.assertTrue(error.Fail())
            self.assertIn('OH NOE', error.GetCString())


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_exceptions_logged(self):
        messages = list()
        self.dbg.SetLoggingCallback(messages.append)
        self.handleCmd('log enable lldb script')
        self.dbg.SetOutputFile(lldb.SBFile(BadIO()))
        self.handleCmd('script 1+1')
        self.assertTrue(any('OH NOE' in msg for msg in messages))


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_flush(self):
        flushed = MutableBool(False)
        closed = MutableBool(False)
        f = FlushTestIO(flushed, closed)
        self.assertFalse(flushed)
        self.assertFalse(closed)
        sbf = lldb.SBFile(f)
        self.assertFalse(flushed)
        self.assertFalse(closed)
        sbf = None
        self.assertFalse(flushed)
        self.assertTrue(closed)
        self.assertTrue(f.closed)

        flushed = MutableBool(False)
        closed = MutableBool(False)
        f = FlushTestIO(flushed, closed)
        self.assertFalse(flushed)
        self.assertFalse(closed)
        sbf = lldb.SBFile.Create(f, borrow=True)
        self.assertFalse(flushed)
        self.assertFalse(closed)
        sbf = None
        self.assertTrue(flushed)
        self.assertFalse(closed)
        self.assertFalse(f.closed)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_fileno_flush(self):
        with open(self.out_filename, 'w') as f:
            f.write("foo")
            sbf = lldb.SBFile(f)
            sbf.Write(b'bar')
            sbf = None
            self.assertTrue(f.closed)
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read(), 'foobar')

        with open(self.out_filename, 'w+') as f:
            f.write("foo")
            sbf = lldb.SBFile.Create(f, borrow=True)
            sbf.Write(b'bar')
            sbf = None
            self.assertFalse(f.closed)
            f.seek(0)
            self.assertEqual(f.read(), 'foobar')


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_close(self):
        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetOutputFile(f)
            self.assertTrue(status.Success())
            self.handleCmd("help help", check=False, collect_result=False)
            # make sure the file wasn't closed early.
            f.write("\nZAP\n")
            lldb.SBDebugger.Destroy(self.dbg)
            # check that output file was closed when debugger was destroyed.
            with self.assertRaises(ValueError):
                f.write("\nQUUX\n")
        with open(self.out_filename, 'r') as f:
            output = f.read()
            self.assertTrue(re.search(r'Show a list of all debugger commands', output))
            self.assertTrue(re.search(r'ZAP', output))


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_stdout(self):
        f = io.StringIO()
        status = self.dbg.SetOutputFile(f)
        self.assertTrue(status.Success())
        self.handleCmd(r"script sys.stdout.write('foobar\n')")
        self.assertEqual(f.getvalue().strip().split(), ["foobar", "7"])


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_stdout_file(self):
        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetOutputFile(f)
            self.assertTrue(status.Success())
            self.handleCmd(r"script sys.stdout.write('foobar\n')")
        with open(self.out_filename, 'r') as f:
            # In python2 sys.stdout.write() returns None, which
            # the REPL will ignore, but in python3 it will
            # return the number of bytes written, which the REPL
            # will print out.
            lines = [x for x in f.read().strip().split() if x != "7"]
            self.assertEqual(lines, ["foobar"])


    @skipIf(py_version=['<', (3,)])
    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_identity(self):

        f = io.StringIO()
        sbf = lldb.SBFile(f)
        self.assertTrue(f is sbf.GetFile())
        sbf.Close()
        self.assertTrue(f.closed)

        f = io.StringIO()
        sbf = lldb.SBFile.Create(f, borrow=True)
        self.assertTrue(f is sbf.GetFile())
        sbf.Close()
        self.assertFalse(f.closed)

        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile(f)
            self.assertTrue(f is sbf.GetFile())
            sbf.Close()
            self.assertTrue(f.closed)

        with open(self.out_filename, 'w') as f:
            sbf = lldb.SBFile.Create(f, borrow=True)
            self.assertFalse(f is sbf.GetFile())
            sbf.Write(b"foobar\n")
            self.assertEqual(f.fileno(), sbf.GetFile().fileno())
            sbf.Close()
            self.assertFalse(f.closed)

        with open(self.out_filename, 'r') as f:
            self.assertEqual("foobar", f.read().strip())

        with open(self.out_filename, 'wb') as f:
            sbf = lldb.SBFile.Create(f, borrow=True, force_io_methods=True)
            self.assertTrue(f is sbf.GetFile())
            sbf.Write(b"foobar\n")
            self.assertEqual(f.fileno(), sbf.GetFile().fileno())
            sbf.Close()
            self.assertFalse(f.closed)

        with open(self.out_filename, 'r') as f:
            self.assertEqual("foobar", f.read().strip())

        with open(self.out_filename, 'wb') as f:
            sbf = lldb.SBFile.Create(f, force_io_methods=True)
            self.assertTrue(f is sbf.GetFile())
            sbf.Write(b"foobar\n")
            self.assertEqual(f.fileno(), sbf.GetFile().fileno())
            sbf.Close()
            self.assertTrue(f.closed)

        with open(self.out_filename, 'r') as f:
            self.assertEqual("foobar", f.read().strip())


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_back_and_forth(self):
        with open(self.out_filename, 'w') as f:
            # at each step here we're borrowing the file, so we have to keep
            # them all alive until the end.
            sbf = lldb.SBFile.Create(f, borrow=True)
            def i(sbf):
                for i in range(10):
                    f = sbf.GetFile()
                    self.assertEqual(f.mode, "w")
                    yield f
                    sbf = lldb.SBFile.Create(f, borrow=True)
                    yield sbf
                    sbf.Write(str(i).encode('ascii') + b"\n")
            files = list(i(sbf))
        with open(self.out_filename, 'r') as f:
            self.assertEqual(list(range(10)), list(map(int, f.read().strip().split())))


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_set_filehandle_none(self):
        self.assertRaises(Exception, self.dbg.SetOutputFile, None)
        self.assertRaises(Exception, self.dbg.SetOutputFile, "ham sandwich")
        self.assertRaises(Exception, self.dbg.SetOutputFileHandle, "ham sandwich")
        self.assertRaises(Exception, self.dbg.SetInputFile, None)
        self.assertRaises(Exception, self.dbg.SetInputFile, "ham sandwich")
        self.assertRaises(Exception, self.dbg.SetInputFileHandle, "ham sandwich")
        self.assertRaises(Exception, self.dbg.SetErrorFile, None)
        self.assertRaises(Exception, self.dbg.SetErrorFile, "ham sandwich")
        self.assertRaises(Exception, self.dbg.SetErrorFileHandle, "ham sandwich")

        with open(self.out_filename, 'w') as f:
            status = self.dbg.SetOutputFile(f)
            self.assertTrue(status.Success())
            status = self.dbg.SetErrorFile(f)
            self.assertTrue(status.Success())
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)
            sbf = self.dbg.GetOutputFile()
            if sys.version_info.major >= 3:
                # python 2 lacks PyFile_FromFd, so GetFile() will
                # have to duplicate the file descriptor and make a FILE*
                # in order to convert a NativeFile it back to a python
                # file.
                self.assertEqual(sbf.GetFile().fileno(), 1)
            sbf = self.dbg.GetErrorFile()
            if sys.version_info.major >= 3:
                self.assertEqual(sbf.GetFile().fileno(), 2)
        with open(self.out_filename, 'r') as f:
            status = self.dbg.SetInputFile(f)
            self.assertTrue(status.Success())
            self.dbg.SetInputFileHandle(None, False)
            sbf = self.dbg.GetInputFile()
            if sys.version_info.major >= 3:
                self.assertEqual(sbf.GetFile().fileno(), 0)


    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def test_sbstream(self):

        with open(self.out_filename, 'w') as f:
            stream = lldb.SBStream()
            stream.RedirectToFile(f)
            stream.Print("zork")
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), "zork")

        with open(self.out_filename, 'w') as f:
            stream = lldb.SBStream()
            stream.RedirectToFileHandle(f, True)
            stream.Print("Yendor")
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), "Yendor")

        stream = lldb.SBStream()
        f = open(self.out_filename,  'w')
        stream.RedirectToFile(lldb.SBFile.Create(f, borrow=False))
        stream.Print("Frobozz")
        stream = None
        self.assertTrue(f.closed)
        with open(self.out_filename, 'r') as f:
            self.assertEqual(f.read().strip(), "Frobozz")

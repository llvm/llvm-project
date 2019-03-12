import io
import sys
import unittest
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

PY3 = (sys.version_info[0] >= 3)
input_name = 'input' if PY3 else 'raw_input'

from pexpect import FSM

class FSMTestCase(unittest.TestCase):
    def test_run_fsm(self):
        def _input(prompt):
            return "167 3 2 2 * * * 1 - ="
        orig_input = getattr(builtins, input_name)
        orig_stdout = sys.stdout
        setattr(builtins, input_name, _input)
        sys.stdout = sio = (io.StringIO if PY3 else io.BytesIO)()
        
        try:
            FSM.main()
        finally:
            setattr(builtins, input_name, orig_input)
            sys.stdout = orig_stdout
        
        printed = sio.getvalue()
        assert '2003' in printed, printed
        
        
if __name__ == '__main__':
    unittest.main()
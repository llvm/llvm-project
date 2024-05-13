import argparse
import socket
import json
import os
import sys

import use_lldb_suite
from lldbsuite.test.gdbclientutils import *

_description = """\
Implements a fake qemu for testing purposes. The executable program
is not actually run. Instead a very basic mock process is presented
to lldb. This allows us to check the invocation parameters.

The behavior of the emulated "process" is controlled via its command line
arguments, which should take the form of key:value pairs. Currently supported
actions are:
- dump: Dump the state of the emulator as a json dictionary. <value> specifies
  the target filename.
- stdout: Write <value> to program stdout.
- stderr: Write <value> to program stderr.
- stdin: Read a line from stdin and store it in the emulator state. <value>
  specifies the dictionary key.
"""


class MyResponder(MockGDBServerResponder):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def cont(self):
        for a in self._state["args"]:
            action, data = a.split(":", 1)
            if action == "dump":
                with open(data, "w") as f:
                    json.dump(self._state, f)
            elif action == "stdout":
                sys.stdout.write(data)
                sys.stdout.flush()
            elif action == "stderr":
                sys.stderr.write(data)
                sys.stderr.flush()
            elif action == "stdin":
                self._state[data] = sys.stdin.readline()
            else:
                print("Unknown action: %r\n" % a)
                return "X01"
        return "W47"


class FakeEmulator(MockGDBServer):
    def __init__(self, addr, state):
        super().__init__(UnixServerSocket(addr))
        self.responder = MyResponder(state)


def main():
    parser = argparse.ArgumentParser(
        description=_description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-g", metavar="unix-socket", required=True)
    parser.add_argument("-0", metavar="arg0")
    parser.add_argument("-fake-arg", dest="fake-arg")
    parser.add_argument("program", help="The program to 'emulate'.")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    state = vars(args)
    state["environ"] = dict(os.environ)
    emulator = FakeEmulator(args.g, state)
    emulator.run()


if __name__ == "__main__":
    main()

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
import tempfile
from ipykernel.kernelbase import Kernel

__version__ = "0.0.1"


class TableGenKernel(Kernel):
    """Kernel using llvm-tblgen inside jupyter.

    All input is treated as TableGen unless the first non whitespace character
    is "%" in which case it is a "magic" line.

    The only magic line supported is "%args". The rest of the line is the
    arguments passed to llvm-tblgen.

    This is "cell magic" meaning it applies to the whole cell. Therefore
    it must be the first line, or part of a run of magic lines starting
    from the first line.

    ```tablgen
    %args
    %args --print-records --print-detailed-records
    class Stuff {
      string Name;
    }

    def a_thing : Stuff {}
    ```

    """

    implementation = "tablegen"
    implementation_version = __version__

    language_version = __version__
    language = "tablegen"
    language_info = {
        "name": "tablegen",
        "mimetype": "text/x-tablegen",
        "file_extension": ".td",
        "pygments_lexer": "text",
    }

    def __init__(self, **kwargs):
        Kernel.__init__(self, **kwargs)
        self._executable = None

    @property
    def banner(self):
        return "llvm-tblgen kernel %s" % __version__

    @property
    def executable(self):
        """If this is the first run, search for llvm-tblgen.
        Otherwise return the cached path to it."""
        if self._executable is None:
            path = os.environ.get("LLVM_TBLGEN_EXECUTABLE")
            if path is not None and os.path.isfile(path) and os.access(path, os.X_OK):
                self._executable = path
            else:
                path = shutil.which("llvm-tblgen")
                if path is None:
                    raise OSError("llvm-tblgen not found, please see README")
                self._executable = path

        return self._executable

    def get_magic(self, code):
        """Given a block of code remove the magic lines from it and return
        a tuple of the code lines (newline joined) and a list of magic lines
        with their leading spaces removed.

        Currently we only look for "cell magic" which must be at the start of
        the cell. Meaning the first line, or a set of lines beginning with %
        that come before the first non-magic line.

        >>> k.get_magic("")
        ('', [])
        >>> k.get_magic("Hello World.\\nHello again.")
        ('Hello World.\\nHello again.', [])
        >>> k.get_magic("   %foo a b c")
        ('', ['%foo a b c'])
        >>> k.get_magic("   %foo a b c\\n%foo\\nFoo")
        ('Foo', ['%foo a b c', '%foo'])
        >>> k.get_magic("Foo\\n%foo\\nFoo")
        ('Foo\\n%foo\\nFoo', [])
        >>> k.get_magic("%foo\\n   Foo\\n%foo")
        ('   Foo\\n%foo', ['%foo'])
        >>> k.get_magic("%foo\\n\\n%foo")
        ('\\n%foo', ['%foo'])
        >>> k.get_magic("%foo\\n \\n%foo")
        (' \\n%foo', ['%foo'])
        """
        magic_lines = []
        code_lines = []

        lines = code.splitlines()
        while lines:
            line = lines.pop(0)
            possible_magic = line.lstrip()
            if possible_magic.startswith("%"):
                magic_lines.append(possible_magic)
            else:
                code_lines = [line, *lines]
                break

        return "\n".join(code_lines), magic_lines

    def do_execute(
        self, code, silent, store_history=True, user_expressions=None, allow_stdin=False
    ):
        """Execute user code using llvm-tblgen binary."""
        code, magic = self.get_magic(code)

        extra_args = []
        for m in magic:
            if m.startswith("%args"):
                # Last one in wins
                extra_args = m.split()[1:]

        with tempfile.TemporaryFile("w+") as f:
            f.write(code)
            f.seek(0)
            got = subprocess.run(
                [self.executable, *extra_args],
                stdin=f,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )

        if not silent:
            if got.stderr:
                self.send_response(
                    self.iopub_socket, "stream", {"name": "stderr", "text": got.stderr}
                )
            else:
                self.send_response(
                    self.iopub_socket, "stream", {"name": "stdout", "text": got.stdout}
                )

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }


if __name__ == "__main__":
    import doctest

    doctest.testmod(extraglobs={"k": TableGenKernel()})

# RUN: %{python} %s

"""Unit tests for lit.LitConfig."""

import contextlib
import io
import platform
import unittest

from lit.LitConfig import LitConfig


def make_lit_config():
    return LitConfig(
        progname="lit",
        path=[],
        diagnostic_level="note",
        useValgrind=False,
        valgrindLeakCheck=False,
        valgrindArgs=[],
        noExecute=False,
        debug=False,
        isWindows=(platform.system() == "Windows"),
        order="smart",
        params={},
    )


class TestWriteMessage(unittest.TestCase):
    def test_note_survives_getsourcefile_returning_none(self):
        """note() must not crash when the caller's frame has no source file.

        inspect.getsourcefile() returns None when the calling frame's source is
        not on disk and not in linecache (e.g. lit packaged into a zip/par).
        _write_message() then used to do os.path.abspath(None), raising a
        TypeError and turning an informational note into a fatal error. Simulate
        that frame by exec'ing a note() call compiled with a filename that does
        not exist on disk.
        """
        lit_config = make_lit_config()

        # co_filename points at a path that is not on disk and not in
        # linecache, so inspect.getsourcefile() on this frame returns None.
        fake_filename = "/nonexistent/packaged/lit.cfg.py"
        code = compile(
            "lit_config.note('a note from a frame with no source file')",
            fake_filename,
            "exec",
        )

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            # Must not raise TypeError: expected str, ... not NoneType.
            exec(code, {"lit_config": lit_config})

        # The message is still emitted, tagged with the frame's co_filename.
        output = captured.getvalue()
        self.assertIn(fake_filename, output)
        self.assertIn("note:", output)


if __name__ == "__main__":
    unittest.main()

from lldbsuite.test.lldbinline import CommandParser
from lldbsuite.test.lldbtest import Base
import textwrap


class TestCommandParser(Base):
    def test_indentation(self):
        """Test indentation handling"""
        filename = self.getBuildArtifact("test_file.cpp")
        with open(filename, "w") as f:
            f.write(
                textwrap.dedent(
                    """\
                    int q;
                    int w; //% first break
                    int e;
                    int r; //% second break
                    //% continue second
                    //%   continuing indented
                      //% not indented
                    int t; //% third break
                    """
                )
            )
        p = CommandParser()
        p.parse_source_files([filename])

        def bkpt(line, cmd):
            return {"file_name": filename, "line_number": line, "command": cmd}

        self.assertEqual(
            p.breakpoints,
            [
                bkpt(2, "first break"),
                bkpt(
                    4,
                    "second break\ncontinue second\n  continuing indented\nnot indented",
                ),
                bkpt(8, "third break"),
            ],
        )

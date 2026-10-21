# RUN: %{python} %s

"""Unit tests for lit.util.printHistogram."""

import contextlib
import io
import unittest

from lit.util import printHistogram


class TestPrintHistogram(unittest.TestCase):
    def test_all_zero_elapsed(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            printHistogram(
                [
                    ("time-tests :: a.txt", 0.0),
                    ("time-tests :: b.txt", 0.0),
                    ("time-tests :: c.txt", 0.0),
                ],
                "all",
                title="Tests",
            )
        bar = "[                                        ]"
        filled = "[****************************************]"
        self.assertEqual(
            buf.getvalue(),
            "Slowest Tests (3 of 3):\n"
            "--------------------------------------------------------------------------\n"
            "0.00s: time-tests :: c.txt\n"
            "0.00s: time-tests :: b.txt\n"
            "0.00s: time-tests :: a.txt\n"
            "\n"
            "Test Times (3):\n"
            "--------------------------------------------------------------------------\n"
            "[    Range    ] :: [               Percentage               ] :: [Count]\n"
            "--------------------------------------------------------------------------\n"
            f"[0.500s,0.550s) :: {bar} :: [0/3]\n"
            f"[0.450s,0.500s) :: {bar} :: [0/3]\n"
            f"[0.400s,0.450s) :: {bar} :: [0/3]\n"
            f"[0.350s,0.400s) :: {bar} :: [0/3]\n"
            f"[0.300s,0.350s) :: {bar} :: [0/3]\n"
            f"[0.250s,0.300s) :: {bar} :: [0/3]\n"
            f"[0.200s,0.250s) :: {bar} :: [0/3]\n"
            f"[0.150s,0.200s) :: {bar} :: [0/3]\n"
            f"[0.100s,0.150s) :: {bar} :: [0/3]\n"
            f"[0.050s,0.100s) :: {bar} :: [0/3]\n"
            f"[0.000s,0.050s) :: {filled} :: [3/3]\n"
            "--------------------------------------------------------------------------\n",
        )


if __name__ == "__main__":
    unittest.main()

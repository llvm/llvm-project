#!/usr/bin/env python
"""Mock GoogleTest executable for testing lit GoogleTest format.

Simulates GoogleTest discovery (--gtest_list_tests) and execution
with JSON output. Used by googletest-empty-subdir.py test.
"""

import os
import sys

# Number of parts to split GTEST_OUTPUT format ("json:/path/to/file.json")
# Split at first colon only to handle paths with colons (e.g., Windows C:\\path)
PROTOCOL_MAX_SPLIT = 1

if len(sys.argv) == 3 and sys.argv[1] == "--gtest_list_tests":
    if sys.argv[2] != "--gtest_filter=-*DISABLED_*":
        raise ValueError(f"unexpected argument: {sys.argv[2]}")
    print(
        """\
FirstTest.
  subTestA
  subTestB
SecondTest.
  subTestC"""
    )
    sys.exit(0)
elif len(sys.argv) != 1:
    # sharding and json output are specified using environment variables
    raise ValueError(f"unexpected argument: {' '.join(sys.argv[1:])!r}")

if "GTEST_OUTPUT" not in os.environ:
    raise ValueError("missing environment variable: GTEST_OUTPUT")

if not os.environ["GTEST_OUTPUT"].startswith("json:"):
    raise ValueError(f"must emit json output: {os.environ['GTEST_OUTPUT']}")

output = """\
{
"random_seed": 123,
"testsuites": [
    {
        "name": "FirstTest",
        "testsuite": [
            {
                "name": "subTestA",
                "result": "COMPLETED",
                "time": "0.001s"
            },
            {
                "name": "subTestB",
                "result": "COMPLETED",
                "time": "0.001s",
                "failures": [
                    {
                        "failure": "Test intentionally fails",
                        "type": ""
                    }
                ]
            }
        ]
    },
    {
        "name": "SecondTest",
        "testsuite": [
            {
                "name": "subTestC",
                "result": "COMPLETED",
                "time": "0.001s"
            }
        ]
    }
]
}"""

json_filename = os.environ["GTEST_OUTPUT"].split(":", PROTOCOL_MAX_SPLIT)[1]
with open(json_filename, "w", encoding="utf-8") as f:
    print("[ RUN      ] FirstTest.subTestA", flush=True)
    print("[       OK ] FirstTest.subTestA (1 ms)", flush=True)
    print("[ RUN      ] FirstTest.subTestB", flush=True)
    print("Test intentionally fails", file=sys.stderr, flush=True)
    print("[  FAILED  ] FirstTest.subTestB (1 ms)", flush=True)
    print("[ RUN      ] SecondTest.subTestC", flush=True)
    print("[       OK ] SecondTest.subTestC (1 ms)", flush=True)
    f.write(output)

sys.exit(1)

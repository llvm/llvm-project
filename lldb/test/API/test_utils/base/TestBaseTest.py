"""
Test TestBase test functions.
"""

import io

from lldbsuite.test.lldbtest import *
from lldbsuite.test_event import build_exception

class TestBuildMethod(Base):

    def setUp(self):
        super().setUp()
        self._traces = []
        self.traceAlways = True

    # override the parent trace method
    def trace(self, *args, **kwargs):
        buf = io.StringIO()
        print(*args, file=buf, **kwargs)
        self._traces.append(buf.getvalue())

    def test_build_fails_helpfully(self):
        try:
            self.build(dictionary={"CXX_SOURCES": "nonexisting-file.cpp"})
        except build_exception.BuildError as e:
            self.assertIn("nonexisting-file.cpp", str(e))
        else:
            self.fail("BuildError not raised!")

    def test_build_logs_traces(self):
        self.build(dictionary={"CXX_SOURCES": "return0.cpp"})
        self.assertIn("CXX_SOURCES", self._traces[0])
        self.assertIn("return0.o", self._traces[1])

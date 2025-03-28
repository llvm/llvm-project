import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    @skipUnlessFoundation
    def test_actor_unprioritised_jobs(self):
        """Verify that an actor exposes its unprioritised jobs (queue)."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.GetSelectedFrame()
        unprioritised_jobs = frame.var("a.$defaultActor.unprioritised_jobs")
        # There are 4 child tasks (async let), the first one occupies the actor
        # with a sleep, the next 3 go on to the queue.
        self.assertEqual(unprioritised_jobs.num_children, 3)
        for job in unprioritised_jobs:
            self.assertRegex(job.name, r"^\d+")
            self.assertRegex(job.summary, r"^id:\d+ flags:\S+")

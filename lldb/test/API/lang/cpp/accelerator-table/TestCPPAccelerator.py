import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPAcceleratorTableTestCase(TestBase):

    @skipUnlessDarwin
    @skipIf(debug_info=no_match(["dwarf"]))
    def test(self):
        """Test that type lookups fail early (performance)"""
        self.build()

        logfile = self.getBuildArtifact('dwarf.log')

        self.expect('log enable dwarf lookups -f' + logfile)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.cpp'))
        # Pick one from the middle of the list to have a high chance
        # of it not being in the first file looked at.
        self.expect('frame variable inner_d')

        with open(logfile) as f:
            log = f.readlines()
        n = 0
        for line in log:
            if re.findall(r'[abcdefg]\.o: FindByNameAndTag\(\)', line):
                self.assertIn("d.o", line)
                n += 1

        self.assertEqual(n, 1, "".join(log))

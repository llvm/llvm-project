from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftTypeAlias(TestBase):

    @swiftTest
    def test(self):
        """Test type aliases are only searched in the debug info once"""
        self.build()
        log = self.getBuildArtifact("dwarf.log")
        self.expect("log enable dwarf lookups -f " + log)

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"), extra_images=['Dylib'])
        self.expect("target variable foo", substrs=["(Dylib.MyAlias)", "23"])
        self.expect("target variable bar",
                    substrs=["(Dylib.MyGenericAlias<Dylib.MyAlias>)", "42"])

        import io
        logfile = io.open(log, "r", encoding='utf-8')
        foo_lookups = 0
        bar_lookups = 0
        for line in logfile:
            if ' SymbolFileDWARF::FindTypes (sc, name="$s5Dylib7MyAliasaD"' in line:
                foo_lookups += 1

            if ' SymbolFileDWARF::FindTypes (sc, name="$s5Dylib14MyGenericAliasaD' in line:
                bar_lookups += 1
        self.assertEquals(foo_lookups, 0)
        # FIXME: This does not actually work yet, it should also be 0.
        #        We look for Dylib.MyGenericAlias
        #        but we have Dylib.MyGenericAlias<Dylib.MyAlias> in the debug info.
        self.assertGreater(bar_lookups, 1)

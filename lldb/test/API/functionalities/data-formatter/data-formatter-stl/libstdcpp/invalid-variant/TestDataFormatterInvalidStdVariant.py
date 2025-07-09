import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class InvalidStdVariantDataFormatterTestCase(TestBase):
    @add_test_categories(["libstdcxx"])
    def test(self):
        """Test STL data formatters for std::variant with invalid index."""
        self.build()

        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect(
            "frame variable v1",
            substrs=["v1 =  Active Type = char  {", "Value = 'x'", "}"],
        )

        var_v1 = thread.frames[0].FindVariable("v1")
        var_v1_raw_obj = var_v1.GetNonSyntheticValue()
        index_obj = var_v1_raw_obj.GetChildMemberWithName("_M_index")
        self.assertTrue(index_obj and index_obj.IsValid())

        INVALID_INDEX = "100"
        index_obj.SetValueFromCString(INVALID_INDEX)

        self.expect("frame variable v1", substrs=["v1 =  <Invalid>"])

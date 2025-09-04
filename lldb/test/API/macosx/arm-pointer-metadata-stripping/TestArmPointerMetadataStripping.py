import lldb
import json
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
@skipIf(archs=no_match(["arm64", "arm64e"]))
class TestArmPointerMetadataStripping(TestBase):
    # Use extra_symbols.json as a template to add a new symbol whose address
    # contains non-zero high order bits set.
    def create_symbols_file(self):
        template_path = os.path.join(self.getSourceDir(), "extra_symbols.json")
        with open(template_path, "r") as f:
            symbols_data = json.load(f)

        target = self.dbg.GetSelectedTarget()
        symbols_data["triple"] = target.GetTriple()

        module = target.GetModuleAtIndex(0)
        symbols_data["uuid"] = module.GetUUIDString()

        json_filename = self.getBuildArtifact("extra_symbols.json")
        with open(json_filename, "w") as file:
            json.dump(symbols_data, file, indent=4)

        return json_filename

    def test(self):
        self.build()
        src = lldb.SBFileSpec("main.c")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", src
        )

        symbols_file = self.create_symbols_file()
        self.runCmd(f"target module add {symbols_file}")

        # The high order bits should be stripped.
        self.expect_expr("get_high_bits(&myglobal_json)", result_value="0")

        # Mark all bits as used for addresses and ensure bits are no longer stripped.
        self.runCmd("settings set target.process.virtual-addressable-bits 64")
        self.expect_expr(
            "get_high_bits(&myglobal_json)", result_value=str(0x1200000000000000)
        )

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def make_json_objfile(outpath, triple):
    '''
    Create a JSON object file at `outpath` for the given `triple` that can be
    used to test the "target hook add" where
    '''
    data = {
        "triple": triple,
        "uuid": "ED6F6CBD-7357-3D32-B3E9-FDA98E07D863",
        "type": "sharedlibrary",
        "sections": [
            {
                "user_id": 0x200,
                "name": ".text",
                "address": 0,
                "size": 0x1000,
                "flags": 0x202,
                "file_offset": 0,
                "file_size": 0,
                "read": True,
                "write": False,
                "execute": True,
            }
        ],
        "symbols": [
            {
                "name": "foo",
                "type": "code",
                "address": 0x0,
                "size": 0x100,
            },
            {
                "name": "bar",
                "type": "code",
                "address": 0x100,
                "size": 0x200,
            },
            {
                "name": "baz",
                "type": "code",
                "address": 0x300,
                "size": 0x300,
            },
        ],
    }
    with open(outpath, "w") as file:
        json.dump(data, file, indent=4)
        return True
    return False


class TestCase(TestBase):
    @no_debug_info_test
    def test_resolve_addr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        addrhook_python_path = self.getSourcePath("addrhook.py")
        addrhook_module_path = self.getBuildArtifact("addrhook.json")
        target = self.dbg.CreateTarget(exe)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set a breakpoint here", lldb.SBFileSpec("main.cpp", False)
        )

        # We need to find a memory region that has no permissions that doesn't
        # start at 0x0 and is at least 0x1000 bytes in size. We'll use this
        # region to load our JSON object file to ensure it doesn't collide with
        # any other modules in the target. We'll use the base address of this
        # region as the base address for our JSON object file.
        region_to_use = None
        curr_addr = 0x0
        while True:
            region = lldb.SBMemoryRegionInfo()
            error = target.process.GetMemoryRegionInfo(curr_addr, region)
            if error.Fail():
                print(f'error: "{error}"')
                break
            base_addr = region.GetRegionBase()
            end_addr = region.GetRegionEnd()
            print(f"Found memory region: [{base_addr:#x}-{end_addr:#x})")
            # We don't want a memory region that starts at 0x0, invalid ranges
            # and any regions that are too small.
            if (base_addr > 0x0) and ((end_addr - base_addr) >= 0x1000):
                if not region.IsReadable() and not region.IsWritable() and not region.IsExecutable():
                    print(f"Found a memory region with no permissions at {region}")
                    region_to_use = region
                    break
            if end_addr == 0xffffffffffffffff:
                break
            curr_addr = end_addr

        # If we didn't find a memory region with no permissions, then skip the
        # test.
        if region_to_use is None:
            return self.skipTest("No memory region with no permissions found")

        json_module_load_addr = region_to_use.GetRegionBase()
        # Load the target hook script file.
        self.dbg.HandleCommand(f'command script import "{addrhook_python_path}"')
        # Setup the "target hook" to use the addrhook.py script and pass in the
        # path to the JSON object file and the base address to load it at as
        # extra arguments to the script.
        self.dbg.HandleCommand(f'target hook add --script-class addrhook.Hooks -k path -v "{addrhook_module_path}" -k base_addr -v {json_module_load_addr:#x}')

        # Verify we can resolve addresses for "foo", "bar" and "baz" symbols in
        # the JSON object file.
        addr = target.ResolveLoadAddress(json_module_load_addr + 0x0)
        self.assertTrue(addr.IsValid(), "Address should be valid")
        self.assertTrue(addr.GetSection().IsValid(), f"Address ({addr}) should have a valid section")
        self.assertEqual(addr.GetSection().GetName(), ".text", "Address should be in .text section")
        self.assertEqual(addr.GetSymbol().GetName(), "foo", "Address should resolve to symbol 'foo'")

        addr = target.ResolveLoadAddress(json_module_load_addr + 0x100)
        self.assertTrue(addr.IsValid(), "Address should be valid")
        self.assertTrue(addr.GetSection().IsValid(), "Address should have a valid section")
        self.assertEqual(addr.GetSection().GetName(), ".text", "Address should be in .text section")
        self.assertEqual(addr.GetSymbol().GetName(), "bar", "Address should resolve to symbol 'bar'")

        addr = target.ResolveLoadAddress(json_module_load_addr + 0x300)
        self.assertTrue(addr.IsValid(), "Address should be valid")
        self.assertTrue(addr.GetSection().IsValid(), "Address should have a valid section")
        self.assertEqual(addr.GetSection().GetName(), ".text", "Address should be in .text section")
        self.assertEqual(addr.GetSymbol().GetName(), "baz", "Address should resolve to symbol 'baz'")

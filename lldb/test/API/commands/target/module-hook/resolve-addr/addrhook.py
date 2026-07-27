import lldb
import json

from lldb.plugins.scripted_hook import ScriptedHook

# Make a JSON object file at `outpath` for the given `triple` that can be
# used to test the "target hook add" where the target hook will be called to
# resolve an address that is not in any of the target's modules.
def make_json_objfile(outpath, triple):
    data = {
        "triple": triple,
        "uuid": "ED6F6CBD-7357-3D32-B3E9-FDA98E07D863",
        "type": "sharedlibrary",
        "sections": [
            {
                "user_id": 0x200,
                "name": ".text",
                "address": 0x0,
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

class Hooks(ScriptedHook):
    def __init__(self, target, extra_args, internal_dict):
        super().__init__(target, extra_args)
        print(f'extra_args type is "{type(extra_args)}"')
        self.path = str(extra_args.GetValueForKey("path"))
        self.base_addr = int(extra_args.GetValueForKey("base_addr"))
        print(f'self.path = "{self.path}"')
        print(f'self.base_addr = {self.base_addr:#x}')

    def handle_stop(self,
                    exe_ctx: lldb.SBExecutionContext,
                    stream: lldb.SBStream) -> bool:
        # This method is required to be implemented. Return true to stop the
        # process, or false to continue.
        return True  # Stop

    def handle_resolve_addr(self,
                            load_addr: int,
                            stream: lldb.SBStream) -> lldb.SBAddress:
        '''
        Called when the target hook needs to resolve an address when the target
        was not able to resolve the address. We will create a JSON object
        file and load it into the target, then resolve the address in this JSON
        object file.
        '''
        module_load_addr = self.base_addr
        module_load_addr_end = self.base_addr + 0x1000
        if module_load_addr <= load_addr and load_addr < module_load_addr_end:
            if self.target.module['addrhook.json'] is None:
                if make_json_objfile(self.path, self.target.triple):
                    print(f"Successfully created JSON object file at '{self.path}'")
                    module = self.target.AddModule(self.path, None, None)
                    if module:
                        print(f"Module loaded successfully, setting load address for .text section to {module_load_addr:#x}")
                        self.target.SetSectionLoadAddress(
                            module.FindSection(".text"),
                            module_load_addr)
                addr = self.target.ResolveLoadAddress(load_addr)
                print(f"Resolved address {load_addr:#x} to {addr}")
                return addr
        return lldb.SBAddress()  # Return an invalid address if we didn't resolve it

import lldb
import os
import binascii
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

MODULE_ID = 4
LOAD_ADDRESS = MODULE_ID << 32
WASM_LOCAL_ADDR = 0x103E0


def format_register_value(val):
    """
    Encode each byte by two hex digits in little-endian order.
    """
    result = ""
    mask = 0xFF
    shift = 0
    for i in range(0, 8):
        x = (val & mask) >> shift
        result += format(x, "02x")
        mask <<= 8
        shift += 8
    return result


class WasmStackFrame:
    def __init__(self, address):
        self._address = address

    def __str__(self):
        return format_register_value(LOAD_ADDRESS | self._address)


class WasmCallStack:
    def __init__(self, wasm_stack_frames):
        self._wasm_stack_frames = wasm_stack_frames

    def __str__(self):
        result = ""
        for frame in self._wasm_stack_frames:
            result += str(frame)
        return result


class FakeMemory:
    def __init__(self, start_addr, end_addr):
        self._base_addr = start_addr
        self._memory = bytearray(end_addr - start_addr)
        self._memoryview = memoryview(self._memory)

    def store_bytes(self, addr, bytes_obj):
        assert addr > self._base_addr
        assert addr < self._base_addr + len(self._memoryview)
        offset = addr - self._base_addr
        chunk = self._memoryview[offset : offset + len(bytes_obj)]
        for i in range(len(bytes_obj)):
            chunk[i] = bytes_obj[i]

    def get_bytes(self, addr, length):
        assert addr > self._base_addr
        assert addr < self._base_addr + len(self._memoryview)

        offset = addr - self._base_addr
        return self._memoryview[offset : offset + length]

    def contains(self, addr):
        return addr - self._base_addr < len(self._memoryview)


class MyResponder(MockGDBServerResponder):
    current_pc = LOAD_ADDRESS | 0x01AD

    def __init__(self, obj_path, module_name="", wasm_call_stacks=[], memory=None):
        self._obj_path = obj_path
        self._module_name = module_name or obj_path
        self._wasm_call_stacks = wasm_call_stacks
        self._call_stack_request_count = 0
        self._memory = memory
        MockGDBServerResponder.__init__(self)

    def respond(self, packet):
        if packet[0:13] == "qRegisterInfo":
            return self.qRegisterInfo(packet[13:])
        if packet.startswith("qWasmCallStack"):
            return self.qWasmCallStack()
        if packet.startswith("qWasmLocal"):
            return self.qWasmLocal(packet)
        return MockGDBServerResponder.respond(self, packet)

    def qSupported(self, client_supported):
        return "qXfer:libraries:read+;PacketSize=1000;vContSupported-"

    def qHostInfo(self):
        return ""

    def QEnableErrorStrings(self):
        return ""

    def qfThreadInfo(self):
        return "m1,"

    def qRegisterInfo(self, index):
        if index == 0:
            return "name:pc;alt-name:pc;bitsize:64;offset:0;encoding:uint;format:hex;set:General Purpose Registers;gcc:16;dwarf:16;generic:pc;"
        return "E45"

    def qProcessInfo(self):
        return "pid:1;ppid:1;uid:1;gid:1;euid:1;egid:1;name:%s;triple:%s;ptrsize:4" % (
            hex_encode_bytes("lldb"),
            hex_encode_bytes("wasm32-unknown-unknown-wasm"),
        )

    def haltReason(self):
        return "T02thread:1;"

    def readRegister(self, register):
        return format_register_value(self.current_pc)

    def qXferRead(self, obj, annex, offset, length):
        if obj == "libraries":
            xml = (
                '<library-list><library name="%s"><section address="%d"/></library></library-list>'
                % (self._module_name, LOAD_ADDRESS)
            )
            return xml, False
        else:
            return None, False

    def readMemory(self, addr, length):
        if self._memory and self._memory.contains(addr):
            chunk = self._memory.get_bytes(addr, length)
            return chunk.hex()
        if addr < LOAD_ADDRESS:
            return "E02"
        result = ""
        with open(self._obj_path, mode="rb") as file:
            file_content = bytearray(file.read())
            if addr >= LOAD_ADDRESS + len(file_content):
                return "E03"
            addr_from = addr - LOAD_ADDRESS
            addr_to = addr_from + min(length, len(file_content) - addr_from)
            for i in range(addr_from, addr_to):
                result += format(file_content[i], "02x")
            file.close()
        return result

    def setBreakpoint(self, packet):
        bp_data = packet[1:].split(",")
        self._bp_address = bp_data[1]
        return "OK"

    def qfThreadInfo(self):
        return "m1"

    def cont(self):
        # Continue execution. Simulates running the Wasm engine until a breakpoint is hit.
        return (
            "T05thread-pcs:"
            + format(int(self._bp_address, 16) & 0x3FFFFFFFFFFFFFFF, "x")
            + ";thread:1"
        )

    def qWasmCallStack(self):
        if len(self._wasm_call_stacks) == 0:
            return ""
        result = str(self._wasm_call_stacks[self._call_stack_request_count])
        self._call_stack_request_count += 1
        return result

    def qWasmLocal(self, packet):
        # Format: qWasmLocal:frame_index;index
        data = packet.split(":")
        data = data[1].split(";")
        frame_index, local_index = data
        if frame_index == "0" and local_index == "2":
            return format_register_value(WASM_LOCAL_ADDR)
        return "E03"


class TestWasm(GDBRemoteTestBase):
    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_load_module_with_embedded_symbols_from_remote(self):
        """Test connecting to a WebAssembly engine via GDB-remote and loading a Wasm module with embedded DWARF symbols"""

        yaml_path = "test_wasm_embedded_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder(obj_path, "test_wasm")

        target = self.dbg.CreateTarget("")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        num_modules = target.GetNumModules()
        self.assertEqual(1, num_modules)

        module = target.GetModuleAtIndex(0)
        num_sections = module.GetNumSections()
        self.assertEqual(5, num_sections)

        code_section = module.GetSectionAtIndex(0)
        self.assertEqual("code", code_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | code_section.GetFileOffset(),
            code_section.GetLoadAddress(target),
        )

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEqual(".debug_info", debug_info_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | debug_info_section.GetFileOffset(),
            debug_info_section.GetLoadAddress(target),
        )

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEqual(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | debug_abbrev_section.GetFileOffset(),
            debug_abbrev_section.GetLoadAddress(target),
        )

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEqual(".debug_line", debug_line_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | debug_line_section.GetFileOffset(),
            debug_line_section.GetLoadAddress(target),
        )

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEqual(".debug_str", debug_str_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | debug_line_section.GetFileOffset(),
            debug_line_section.GetLoadAddress(target),
        )

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_load_module_with_stripped_symbols_from_remote(self):
        """Test connecting to a WebAssembly engine via GDB-remote and loading a Wasm module with symbols stripped into a separate Wasm file"""

        sym_yaml_path = "test_sym.yaml"
        sym_yaml_base, ext = os.path.splitext(sym_yaml_path)
        sym_obj_path = self.getBuildArtifact(sym_yaml_base) + ".wasm"
        self.yaml2obj(sym_yaml_path, sym_obj_path)

        yaml_path = "test_wasm_external_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base) + ".wasm"
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder(obj_path, "test_wasm")

        folder, _ = os.path.split(obj_path)
        self.runCmd(
            "settings set target.debug-file-search-paths " + os.path.abspath(folder)
        )

        target = self.dbg.CreateTarget("")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        num_modules = target.GetNumModules()
        self.assertEqual(1, num_modules)

        module = target.GetModuleAtIndex(0)
        num_sections = module.GetNumSections()
        self.assertEqual(5, num_sections)

        code_section = module.GetSectionAtIndex(0)
        self.assertEqual("code", code_section.GetName())
        self.assertEqual(
            LOAD_ADDRESS | code_section.GetFileOffset(),
            code_section.GetLoadAddress(target),
        )

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEqual(".debug_info", debug_info_section.GetName())
        self.assertEqual(
            lldb.LLDB_INVALID_ADDRESS, debug_info_section.GetLoadAddress(target)
        )

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEqual(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEqual(
            lldb.LLDB_INVALID_ADDRESS, debug_abbrev_section.GetLoadAddress(target)
        )

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEqual(".debug_line", debug_line_section.GetName())
        self.assertEqual(
            lldb.LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target)
        )

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEqual(".debug_str", debug_str_section.GetName())
        self.assertEqual(
            lldb.LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target)
        )

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_simple_wasm_debugging_session(self):
        """Test connecting to a WebAssembly engine via GDB-remote, loading a
        Wasm module with embedded DWARF symbols, setting a breakpoint and
        checking the debuggee state"""

        # simple.yaml was created by compiling simple.c to wasm and using
        # obj2yaml on the output.
        #
        # $ clang -target wasm32 -nostdlib -Wl,--no-entry -Wl,--export-all -O0 -g -o simple.wasm simple.c
        # $ obj2yaml simple.wasm -o simple.yaml
        yaml_path = "simple.yaml"
        yaml_base, _ = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        # Create a fake call stack.
        call_stacks = [
            WasmCallStack(
                [WasmStackFrame(0x019C), WasmStackFrame(0x01E5), WasmStackFrame(0x01FE)]
            ),
        ]

        # Create fake memory for our wasm locals.
        self.memory = FakeMemory(0x10000, 0x20000)
        self.memory.store_bytes(
            WASM_LOCAL_ADDR,
            bytes.fromhex(
                "0000000000000000020000000100000000000000020000000100000000000000"
            ),
        )

        self.server.responder = MyResponder(
            obj_path, "test_wasm", call_stacks, self.memory
        )

        target = self.dbg.CreateTarget("")
        breakpoint = target.BreakpointCreateByName("add")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and location.IsEnabled(), VALID_BREAKPOINT_LOCATION)

        num_modules = target.GetNumModules()
        self.assertEqual(1, num_modules)

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())

        # Check that our frames match our fake call stack.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid())
        self.assertEqual(frame0.GetPC(), LOAD_ADDRESS | 0x019C)
        self.assertIn("add", frame0.GetFunctionName())

        frame1 = thread.GetFrameAtIndex(1)
        self.assertTrue(frame1.IsValid())
        self.assertEqual(frame1.GetPC(), LOAD_ADDRESS | 0x01E5)
        self.assertIn("main", frame1.GetFunctionName())

        # Check that we can resolve local variables.
        a = frame0.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsUnsigned(), 1)

        b = frame0.FindVariable("b")
        self.assertTrue(b.IsValid())
        self.assertEqual(b.GetValueAsUnsigned(), 2)

import lldb
import binascii
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

LLDB_INVALID_ADDRESS = lldb.LLDB_INVALID_ADDRESS
MODULE_ID = 4
LOAD_ADDRESS = MODULE_ID << 32

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


def make_code_address(module_id, offset):
    return 0x4000000000000000 | (module_id << 32) | offset


class MyResponder(MockGDBServerResponder):
    current_pc = LOAD_ADDRESS + 0x0A

    def __init__(self, obj_path, module_name, wasm_call_stacks=[], memory_view=[]):
        self._obj_path = obj_path
        self._module_name = module_name or obj_path
        self._bp_address = 0
        self._wasm_call_stacks = wasm_call_stacks
        self._call_stack_request_count = 0
        self._memory_view = memory_view
        MockGDBServerResponder.__init__(self)

    def SetCurrentPC(self, address):
        self.current_pc = LOAD_ADDRESS + address

    def respond(self, packet):
        if packet[0:13] == "qRegisterInfo":
            return self.qRegisterInfo(packet[13:])
        if packet.startswith("qWasmCallStack"):
            return self.qWasmCallStack()
        if packet.startswith("qWasmLocal"):
            return self.qWasmLocal(packet)
        if packet.startswith("qWasmMem"):
            return self.qWasmMem(packet)
        return MockGDBServerResponder.respond(self, packet)

    def qSupported(self, client_supported):
        return "qXfer:libraries:read+;PacketSize=1000;vContSupported-"

    def qHostInfo(self):
        return ""

    def QEnableErrorStrings(self):
        return ""

    def qfThreadInfo(self):
        return "OK"

    def qRegisterInfo(self, index):
        if index == "0":
            return "name:pc;alt-name:pc;bitsize:64;offset:0;encoding:uint;format:hex;set:General Purpose Registers;gcc:16;dwarf:16;generic:pc;"
        return "E45"

    def qProcessInfo(self):
        return "pid:1;ppid:1;uid:1;gid:1;euid:1;egid:1;name:%s;triple:%s;ptrsize:4" % (
            hex_encode_bytes("lldb"),
            hex_encode_bytes("wasm32-unknown-unknown-wasm"),
        )

    def haltReason(self):
        return "T05thread-pcs:300000083;thread:1;library:;"

    def readRegister(self, register):
        assert register == 0
        return format_register_value(self.current_pc)

    def qXferRead(self, obj, annex, offset, length):
        if obj == "libraries":
            xml = (
                '<library-list><library name="%s"><section address="%d"/></library></library-list>'
                % (self._module_name, make_code_address(MODULE_ID, 0))
            )
            return xml, False
        else:
            return None, False

    def readMemory(self, addr, length):
        result = ""
        with open(self._obj_path, mode="rb") as file:
            if addr < LOAD_ADDRESS:
                return "E02"
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
        result = self._wasm_call_stacks[self._call_stack_request_count].format()
        self._call_stack_request_count = self._call_stack_request_count + 1
        return result

    def qWasmLocal(self, packet):
        # Format: qWasmLocal:frame_index;index
        data = packet.split(":")
        data = data[1].split(";")
        frame_index = data[0]
        local_index = data[1]
        if frame_index == "0" and local_index == "4":
            return "b0ff0000"
        if frame_index == "1" and local_index == "5":
            return "c0ff0000"
        return "E03"

    def qWasmMem(self, packet):
        # Format: qWasmMem:module_id;addr;len
        data = packet.split(":")
        data = data[1].split(";")
        module_id = data[0]
        addr = int(data[1], 16)
        length = int(data[2])
        if module_id != "4":
            return "E03"
        if addr >= 0xFFB8 and addr < 0x10000:
            chunk = self._memory_view[addr : addr + length]
            return chunk.hex()
        return "E03"


class WasmStackFrame:
    pass

    def __init__(self, module_id, address):
        self._module_id = module_id
        self._address = address

    def format(self):
        return format_register_value(make_code_address(self._module_id, self._address))


class WasmCallStack:
    pass

    def __init__(self, wasm_stack_frames):
        self._wasm_stack_frames = wasm_stack_frames

    def format(self):
        result = ""
        for frame in self._wasm_stack_frames:
            result += frame.format()
        return result


class WasmMemorySpan:
    pass

    def __init__(self, offset, bytes):
        self._offset = offset
        self._bytes = bytes

class TestWasm(GDBRemoteTestBase):
    def connect_to_wasm_engine(self, target):
        """
        Create a process by connecting to the mock GDB server running in a mock WebAssembly engine.
        Includes assertions that the process was successfully created.
        """
        listener = self.dbg.GetListener()
        error = lldb.SBError()
        process = target.ConnectRemote(
            listener, self.server.get_connect_url(), "wasm", error
        )
        self.assertTrue(error.Success(), error.description)
        self.assertTrue(process, PROCESS_IS_VALID)
        return process

    def store_bytes(self, offset, bytes_obj):
        chunk = self.memory_view[offset : offset + len(bytes_obj)]
        for i in range(len(bytes_obj)):
            chunk[i] = bytes_obj[i]

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
        process = self.connect_to_wasm_engine(target)
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
            make_code_address(MODULE_ID, code_section.GetFileOffset()),
            code_section.GetLoadAddress(target),
        )

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEqual(".debug_info", debug_info_section.GetName())
        self.assertEqual(
            make_code_address(MODULE_ID, debug_info_section.GetFileOffset()),
            debug_info_section.GetLoadAddress(target),
        )

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEqual(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEqual(
            make_code_address(MODULE_ID, debug_abbrev_section.GetFileOffset()),
            debug_abbrev_section.GetLoadAddress(target),
        )

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEqual(".debug_line", debug_line_section.GetName())
        self.assertEqual(
            make_code_address(MODULE_ID, debug_line_section.GetFileOffset()),
            debug_line_section.GetLoadAddress(target),
        )

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEqual(".debug_str", debug_str_section.GetName())
        self.assertEqual(
            make_code_address(MODULE_ID, debug_line_section.GetFileOffset()),
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
        process = self.connect_to_wasm_engine(target)
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
            make_code_address(MODULE_ID, code_section.GetFileOffset()),
            code_section.GetLoadAddress(target),
        )

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEqual(".debug_info", debug_info_section.GetName())
        self.assertEqual(
            LLDB_INVALID_ADDRESS, debug_info_section.GetLoadAddress(target)
        )

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEqual(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEqual(
            LLDB_INVALID_ADDRESS, debug_abbrev_section.GetLoadAddress(target)
        )

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEqual(".debug_line", debug_line_section.GetName())
        self.assertEqual(
            LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target)
        )

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEqual(".debug_str", debug_str_section.GetName())
        self.assertEqual(
            LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target)
        )

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_simple_wasm_debugging_session(self):
        """Test connecting to a WebAssembly engine via GDB-remote, loading a Wasm module with embedded DWARF symbols, setting a breakpoint and checking the debuggee state"""

        yaml_path = "calc-cpp-o0.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base) + ".wasm"
        self.yaml2obj(yaml_path, obj_path)

        self.memory = bytearray(65536)
        self.memory_view = memoryview(self.memory)
        self.store_bytes(0xFFB8, bytes.fromhex("d8ff0000"))
        self.store_bytes(0xFFBC, bytes.fromhex("e0ff0000"))
        self.store_bytes(0xFFE0, bytes.fromhex("0000000000000000"))
        self.store_bytes(0xFFE8, bytes.fromhex("0000000000004540"))
        self.store_bytes(0xFFF0, bytes.fromhex("0000000000003640"))
        self.store_bytes(0xFFF8, bytes.fromhex("0000000000003440"))

        call_stacks = [
            WasmCallStack(
                [WasmStackFrame(3, 0x00000083), WasmStackFrame(3, 0x0000000F)]
            ),
            WasmCallStack(
                [
                    WasmStackFrame(4, 0x000002AD),
                    WasmStackFrame(4, 0x0000014A),
                    WasmStackFrame(3, 0x00000083),
                    WasmStackFrame(3, 0x0000000F),
                ]
            ),
        ]
        self.server.responder = MyResponder(
            obj_path, "test_wasm", call_stacks, self.memory_view
        )

        target = self.dbg.CreateTarget("")
        breakpoint = target.BreakpointCreateByLocation("calc.cpp", 9)

        process = self.connect_to_wasm_engine(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # Verify all breakpoint locations are enabled.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and location.IsEnabled(), VALID_BREAKPOINT_LOCATION)

        # Continue execution.
        self.runCmd("c")

        # Verify 1st breakpoint location is hit.
        from lldbsuite.test.lldbutil import get_stopped_thread

        thread = get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )
        frame0 = thread.GetFrameAtIndex(0)
        self.server.responder.SetCurrentPC(0x000002AD)

        # Update Wasm server memory state
        self.store_bytes(0xFFE0, bytes.fromhex("0000000000003440"))

        # We should be in function 'bar'.
        self.assertTrue(frame0.IsValid())
        function_name = frame0.GetFunctionName()
        self.assertIn(
            "Calc::add(Number const&)",
            function_name,
            "Unexpected function name {}".format(function_name),
        )

        # We should be able to evaluate the expression "*this".
        value = frame0.EvaluateExpression("*this")
        self.assertEqual(value.GetTypeName(), "Calc")
        field = value.GetChildAtIndex(0)
        self.assertEqual(field.GetName(), "result_")
        self.assertEqual(field.GetTypeName(), "double")
        self.assertEqual(field.GetValue(), "20")

        # Examine next stack frame.
        self.runCmd("up")
        frame1 = thread.GetSelectedFrame()

        # We should be able to evaluate the expression "v2".
        value = frame1.EvaluateExpression("v2")
        self.assertEqual(value.GetTypeName(), "double")
        self.assertEqual(value.GetValue(), "22")

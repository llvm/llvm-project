import lldb
import os
import binascii
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

# Ids of the instances the fake stub loads. An id is written in base 10, both
# in a packet and in the library list it is reported through, so these read
# differently in base 16, which a single digit would not.
MODULE_ID = 16
SECOND_MODULE_ID = 26

# The address spaces an address can point into, in the bits above the id of the
# instance the address belongs to. The object space holds the module image, so
# that is the space a module is loaded in.
WASM_OBJECT_ADDRESS = 1 << 62
WASM_GLOBAL_ADDRESS = 2 << 62
WASM_ID_MASK = 0x3FFFFFFF

LOAD_ADDRESS = WASM_OBJECT_ADDRESS | (MODULE_ID << 32)
WASM_LOCAL_ADDR = 0x103E0

# The key under which a packet names the module instance whose state it reads.
# The key form is also what tells the two shapes of qWasmGlobal apart, since the
# first field is a global index in one and a frame index in the other.
INSTANCE_KEY = "instance:"

# Globals the fake stub holds, as index -> (size in bytes, value). Every
# instance has an index space of its own, so these are the globals of one
# instance and the same index names a different global in another. An index is
# written in base 10 like an instance id, so one of them reads differently in
# base 16.
WASM_GLOBALS = {0: (4, 0x2A), 1: (8, 0xDEADBEEF), 26: (4, 0x33)}
SECOND_WASM_GLOBALS = {0: (4, 0x1234), 1: (8, 0xFEEDFACE)}

# Bytes of the fake stack frame a frame base points at, holding the values of the
# parameters of "add" at the offsets its DWARF gives them, and above them those
# of the variables of "main". A Wasm stack grows down, so the frame of a caller
# sits above the frame of what it called.
WASM_FRAME_BYTES = bytes.fromhex(
    "0000000000000000020000000100000000000000020000000100000000000000"
)
WASM_CALLER_FRAME_ADDR = WASM_LOCAL_ADDR + 16

# The globals simple_global_frame_base.yaml gives the two functions of the module
# as their frame base. Each function is based on a global of its own, so a read
# of the wrong one does not land in the frame it belongs to.
INNER_FRAME_BASE_GLOBAL_INDEX = 0
OUTER_FRAME_BASE_GLOBAL_INDEX = 1
FRAME_BASE_GLOBALS = {
    INNER_FRAME_BASE_GLOBAL_INDEX: (4, WASM_LOCAL_ADDR),
    OUTER_FRAME_BASE_GLOBAL_INDEX: (4, WASM_CALLER_FRAME_ADDR),
}


class WasmModule:
    """
    A module the fake stub has loaded, together with the globals held by the
    instance it was loaded as.
    """

    def __init__(self, obj_path, name, module_id=MODULE_ID, global_values=None):
        self.obj_path = obj_path
        self.name = name
        self.module_id = module_id
        self.load_address = WASM_OBJECT_ADDRESS | (module_id << 32)
        self.global_values = WASM_GLOBALS if global_values is None else global_values
        self._image = None

    def get_image(self):
        """
        The bytes of the module itself, which is what the stub serves in the
        range the module is loaded at.
        """
        if self._image is None:
            with open(self.obj_path, mode="rb") as file:
                self._image = file.read()
        return self._image

    def encode_global(self, global_index):
        """
        Encode the global at the given index, or an error when this instance
        holds no such global. A global is transferred as a whole value, in
        little-endian order.
        """
        value = self.global_values.get(global_index)
        if value is None:
            return "E03"
        size, val = value
        return val.to_bytes(size, "little").hex()


def global_read_packet(global_index, module_id=MODULE_ID):
    """
    The packet that reads a global from the module instance holding it.
    """
    return f"qWasmGlobal:{global_index};{INSTANCE_KEY}{module_id};"


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
    def __init__(self, address, load_address=LOAD_ADDRESS):
        self._address = address
        self._load_address = load_address

    def __str__(self):
        return format_register_value(self._load_address | self._address)


class WasmCallStack:
    def __init__(self, wasm_stack_frames):
        self._wasm_stack_frames = wasm_stack_frames

    def __len__(self):
        return len(self._wasm_stack_frames)

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

    def __init__(
        self,
        modules,
        wasm_call_stacks=[],
        memory=None,
        supports_instance=True,
    ):
        self._modules = modules
        self._wasm_call_stacks = wasm_call_stacks
        self._call_stack_request_count = 0
        self._reported_frames = 0
        self._memory = memory
        self._supports_instance = supports_instance
        MockGDBServerResponder.__init__(self)

    def other(self, packet):
        if packet.startswith("qWasmCallStack"):
            return self.qWasmCallStack()
        if packet.startswith("qWasmLocal"):
            return self.qWasmLocal(packet)
        if packet.startswith("qWasmGlobal"):
            return self.qWasmGlobal(packet)
        return MockGDBServerResponder.other(self, packet)

    def module_with_id(self, module_id):
        """
        The module the stub loaded with the given id, if it loaded one. Nothing
        about a module is shared with another, so an id that names none has no
        code to read and no globals to hand out.
        """
        for module in self._modules:
            if module.module_id == module_id:
                return module
        return None

    def qWasmGlobal(self, packet):
        """
        Read a global. A client that can name the instance holding it does so
        with the instance suffix, and one that cannot names a frame.

        Format: qWasmGlobal:index;instance:id; or
                qWasmGlobal:frame_index;index
        """
        first, _, rest = packet.split(":", 1)[1].partition(";")

        if rest.startswith(INSTANCE_KEY):
            if not self._supports_instance:
                # A stub is only asked for what it advertised.
                return "E05"
            # The global index space belongs to the instance, so an index only
            # names a global together with the instance it indexes. Another
            # instance's globals answer a different question.
            module = self.module_with_id(int(rest[len(INSTANCE_KEY) :].rstrip(";")))
            if module is None:
                return "E04"
            return module.encode_global(int(first))

        if self._supports_instance:
            # A frame cannot name the instance whose globals are indexed, so a
            # stub that can be given an instance is never given a frame.
            return "E05"
        # A frame index that names none of the frames the stub reported is no
        # scope to read a global through.
        if int(first) >= self._reported_frames:
            return "E06"
        # A frame stands in for the instance it is executing, which with a single
        # loaded module is that module.
        return self._modules[0].encode_global(int(rest))

    def qSupported(self, client_supported):
        response = "qXfer:libraries:read+;PacketSize=1000;vContSupported-"
        if self._supports_instance:
            response += ";qWasmInstance+"
        return response

    def qHostInfo(self):
        return ""

    def QEnableErrorStrings(self):
        return ""

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
            libraries = "".join(
                '<library name="%s"><section address="%d"/></library>'
                % (module.name, module.load_address)
                for module in self._modules
            )
            return "<library-list>" + libraries + "</library-list>", False
        else:
            return None, False

    def readMemory(self, addr, length):
        if self._memory and self._memory.contains(addr):
            chunk = self._memory.get_bytes(addr, length)
            return chunk.hex()
        # A module is loaded in the object space of its instance, so an address
        # in that space is what asks for the module image, and the id the address
        # carries picks the instance whose module answers.
        if addr >> 62 != WASM_OBJECT_ADDRESS >> 62:
            return "E02"
        module = self.module_with_id((addr >> 32) & WASM_ID_MASK)
        if module is None:
            return "E02"
        image = module.get_image()
        offset = addr - module.load_address
        if offset >= len(image):
            return "E03"
        end = offset + min(length, len(image) - offset)
        return image[offset:end].hex()

    def setBreakpoint(self, packet):
        bp_data = packet[1:].split(",")
        self._bp_address = bp_data[1]
        return "OK"

    def qfThreadInfo(self):
        return "m1"

    def cont(self):
        # Continue execution. Simulates running the Wasm stub until a breakpoint is hit.
        # A program counter names the space and the instance it belongs to, as
        # the address the breakpoint was set at does.
        return "T05thread-pcs:" + self._bp_address + ";thread:1"

    def qWasmCallStack(self):
        if len(self._wasm_call_stacks) == 0:
            return ""
        call_stack = self._wasm_call_stacks[self._call_stack_request_count]
        self._call_stack_request_count += 1
        # A frame is only a scope the stub can be asked about once it has
        # reported it.
        self._reported_frames = len(call_stack)
        return str(call_stack)

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
        """Test connecting to a WebAssembly stub via GDB-remote and loading a Wasm module with embedded DWARF symbols"""

        yaml_path = "test_wasm_embedded_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder([WasmModule(obj_path, "test_wasm")])

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
        """Test connecting to a WebAssembly stub via GDB-remote and loading a Wasm module with symbols stripped into a separate Wasm file"""

        sym_yaml_path = "test_sym.yaml"
        sym_yaml_base, ext = os.path.splitext(sym_yaml_path)
        sym_obj_path = self.getBuildArtifact(sym_yaml_base) + ".wasm"
        self.yaml2obj(sym_yaml_path, sym_obj_path)

        yaml_path = "test_wasm_external_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base) + ".wasm"
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder([WasmModule(obj_path, "test_wasm")])

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
        """Test connecting to a WebAssembly stub via GDB-remote, loading a
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
        self.memory.store_bytes(WASM_LOCAL_ADDR, WASM_FRAME_BYTES)

        self.server.responder = MyResponder(
            [WasmModule(obj_path, "test_wasm")], call_stacks, self.memory
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

        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame2.IsValid())

        # Wasm frames need distinct, ordered call frame addresses for StackID to
        # tell an inner frame from an outer one. Without them every frame shares
        # one address and stepping mistakes a step in for a step out.
        self.assertLess(frame0.GetCFA(), frame1.GetCFA())
        self.assertLess(frame1.GetCFA(), frame2.GetCFA())

        # Check that we can resolve local variables.
        a = frame0.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsUnsigned(), 1)

        b = frame0.FindVariable("b")
        self.assertTrue(b.IsValid())
        self.assertEqual(b.GetValueAsUnsigned(), 2)

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_max_backtrace_depth(self):
        """Test that a Wasm backtrace stops at the depth the target sets, so
        that a stack recursing without end is not walked to its end."""

        yaml_path = "simple.yaml"
        yaml_base, _ = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        call_stacks = [
            WasmCallStack(
                [WasmStackFrame(0x019C), WasmStackFrame(0x01E5), WasmStackFrame(0x01FE)]
            ),
        ]
        self.server.responder = MyResponder(
            [WasmModule(obj_path, "test_wasm")], call_stacks
        )

        target = self.dbg.CreateTarget("")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        self.assertTrue(thread.GetFrameAtIndex(0).IsValid())

        # A depth set after a stop still applies to the frames a backtrace has
        # not reported yet, which is when a user lowers it.
        self.runCmd("settings set target.process.thread.max-backtrace-depth 2")

        # The stub reported three frames, and the innermost two are the ones
        # kept.
        self.assertEqual(2, thread.GetNumFrames())
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), LOAD_ADDRESS | 0x019C)
        self.assertEqual(thread.GetFrameAtIndex(1).GetPC(), LOAD_ADDRESS | 0x01E5)
        self.assertFalse(thread.GetFrameAtIndex(2).IsValid())

    def build_wasm_module(self, name, yaml_path="simple.yaml", **module_args):
        """
        Build a Wasm module the fake stub reports under the given name. A
        loaded module is told apart from another by the name it is reported
        under, so every instance has to be reported under a name of its own.
        """
        obj_path = self.getBuildArtifact(name)
        self.yaml2obj(yaml_path, obj_path)
        return WasmModule(obj_path, name, **module_args)

    def connect_to_modules(
        self, modules, call_stacks, supports_instance=True, memory=None
    ):
        """
        Connect to a fake stub holding the given modules and return its target
        and process.
        """
        self.server.responder = MyResponder(
            modules,
            call_stacks,
            memory,
            supports_instance=supports_instance,
        )

        target = self.dbg.CreateTarget("")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        return target, process

    def get_globals_address(self, target, module):
        """
        The address the loaded module gives its globals. Reading through it
        rather than a constructed address checks the encoding the object file
        produces and the one the process decodes against each other.
        """
        image = target.FindModule(lldb.SBFileSpec(module.name))
        self.assertTrue(image.IsValid())
        global_section = image.FindSection("global")
        self.assertTrue(global_section.IsValid())
        globals_addr = global_section.GetLoadAddress(target)
        # A global lives in the global index space of its instance rather than
        # in the space the module itself is loaded in.
        self.assertEqual(globals_addr, WASM_GLOBAL_ADDRESS | (module.module_id << 32))
        return globals_addr

    def connect_to_globals(self, call_stacks, supports_instance=True):
        """
        Connect to a fake stub holding WASM_GLOBALS and return its process
        together with the address its module gives its globals.
        """
        module = self.build_wasm_module("test_wasm")
        target, process = self.connect_to_modules(
            [module], call_stacks, supports_instance
        )
        return process, self.get_globals_address(target, module)

    def connect_to_frame_base_globals(self, supports_instance=True, call_stacks=None):
        """
        Connect to a fake stub stopped in a module whose functions are based on
        a global holding the address of their frame, and return its thread.
        """
        module = self.build_wasm_module(
            "test_wasm",
            yaml_path="simple_global_frame_base.yaml",
            global_values=FRAME_BASE_GLOBALS,
        )

        if call_stacks is None:
            call_stacks = [
                WasmCallStack([WasmStackFrame(0x019C), WasmStackFrame(0x01E5)])
            ]
        memory = FakeMemory(0x10000, 0x20000)
        memory.store_bytes(WASM_LOCAL_ADDR, WASM_FRAME_BYTES)
        _, process = self.connect_to_modules(
            [module],
            call_stacks,
            supports_instance=supports_instance,
            memory=memory,
        )

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        return thread

    def packets_received(self, prefix):
        """
        The packets the client sent that start with the given prefix. An
        assertion on the packets themselves names the ones that make it fail,
        which an assertion on whether there are any does not.
        """
        received = self.server.responder.packetLog.get_received()
        return [packet for packet in received if packet.startswith(prefix)]

    def global_reads_received(self, instance):
        """
        The global reads the client sent that name an instance, or those that
        name a frame instead. Both forms share a packet name, so the suffix is
        what tells them apart.
        """
        return [
            packet
            for packet in self.packets_received("qWasmGlobal:")
            if (INSTANCE_KEY in packet) == instance
        ]

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global(self):
        """Test that a WebAssembly global can be read through the address its
        module gives it, and that a read it cannot serve fails instead of
        returning something plausible."""

        call_stacks = [WasmCallStack([WasmStackFrame(0x019C)])]
        process, globals_addr = self.connect_to_globals(call_stacks)

        # A global is read as a whole value.
        error = lldb.SBError()
        data = process.ReadMemory(globals_addr + 0, 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x2A)

        data = process.ReadMemory(globals_addr + 1, 8, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0xDEADBEEF)

        data = process.ReadMemory(globals_addr + 26, 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x33)

        # A type narrower than the global it is held in reads the low bytes,
        # which is how a char or short global is read.
        data = process.ReadMemory(globals_addr + 0, 1, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x2A)

        # Reading more than a global holds would have to come from somewhere
        # else. The next index is not adjacent storage, so this fails rather
        # than returning whatever is nearby.
        process.ReadMemory(globals_addr + 0, 8, error)
        self.assertFalse(error.Success())
        self.assertIn("4-byte global", error.GetCString())

        # Likewise for a global that does not exist.
        process.ReadMemory(globals_addr + 99, 4, error)
        self.assertFalse(error.Success())

        # A global is named by the instance holding it. MODULE_ID differs from
        # every frame index, so a frame index cannot pass for an instance id.
        self.assertPacketLogReceived(
            [
                global_read_packet(0),
                global_read_packet(1),
                global_read_packet(26),
            ]
        )
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_without_reported_call_stack(self):
        """Test that a WebAssembly global can be read while the stub reports no
        call stack for the instance holding it."""

        process, globals_addr = self.connect_to_globals([])

        # A thread always has a frame, which LLDB makes from the registers the
        # stub reports when the Wasm unwinder contributes none. The program
        # counter of that frame belongs to no module, so it is not a frame a
        # global can be read through.
        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        self.assertEqual(1, thread.GetNumFrames())
        self.assertEqual(MyResponder.current_pc, thread.GetFrameAtIndex(0).GetPC())

        error = lldb.SBError()
        data = process.ReadMemory(globals_addr + 0, 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x2A)

        data = process.ReadMemory(globals_addr + 1, 8, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0xDEADBEEF)

        self.assertPacketLogReceived(
            [
                global_read_packet(0),
                global_read_packet(1),
            ]
        )

        # The stub was asked for a call stack and reported none, so the reads
        # above were served without one. Reading a global through a frame is
        # something else, and is what a stub that cannot be told which instance
        # to read is limited to.
        self.assertNotEqual([], self.packets_received("qWasmCallStack"))
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_from_multiple_instances(self):
        """Test that a global is read from the instance that holds it when more
        than one instance is loaded."""

        module = self.build_wasm_module("test_wasm")
        second_module = self.build_wasm_module(
            "test_wasm_second",
            module_id=SECOND_MODULE_ID,
            global_values=SECOND_WASM_GLOBALS,
        )

        # Only the first instance is executing, so the frame available to read a
        # global through belongs to it and cannot stand in for the other one.
        call_stacks = [WasmCallStack([WasmStackFrame(0x019C)])]
        target, process = self.connect_to_modules([module, second_module], call_stacks)

        # Each instance has to be a module of its own for its globals to be its
        # own. Collapsing the two would leave one of the load addresses unused.
        self.assertEqual(2, target.GetNumModules())

        globals_addr = self.get_globals_address(target, module)
        second_globals_addr = self.get_globals_address(target, second_module)
        self.assertNotEqual(globals_addr, second_globals_addr)

        # The same index in either instance names a global of that instance, and
        # the fake stub refuses any instance it did not load, so neither value
        # can come from the wrong place.
        error = lldb.SBError()
        for addr, global_values in [
            (globals_addr, WASM_GLOBALS),
            (second_globals_addr, SECOND_WASM_GLOBALS),
        ]:
            for index, (size, value) in global_values.items():
                data = process.ReadMemory(addr + index, size, error)
                self.assertSuccess(error)
                self.assertEqual(int.from_bytes(data, "little"), value)

        self.assertPacketLogReceived(
            [
                global_read_packet(0),
                global_read_packet(1),
                global_read_packet(0, SECOND_MODULE_ID),
                global_read_packet(1, SECOND_MODULE_ID),
            ]
        )
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_from_default_instance(self):
        """Test that a global is read from the instance a debug session is
        created for, which a stub reports at id zero."""

        # An id of zero yields the load address a stub reports when it cannot
        # report an instance at all, and an address the running code computed
        # carries no id either, so zero has to name an instance.
        module = self.build_wasm_module("test_wasm", module_id=0)
        self.assertEqual(module.load_address, WASM_OBJECT_ADDRESS)

        call_stacks = [WasmCallStack([WasmStackFrame(0x019C, module.load_address)])]
        target, process = self.connect_to_modules([module], call_stacks)
        globals_addr = self.get_globals_address(target, module)

        error = lldb.SBError()
        data = process.ReadMemory(globals_addr + 0, 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x2A)

        # Zero names that instance rather than standing for no instance, so the
        # read is scoped to it instead of falling back to naming a frame.
        self.assertPacketLogReceived([global_read_packet(0, module.module_id)])
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_variable_located_through_global(self):
        """Test that a variable an outer frame locates through a Wasm global is
        read from the instance the frame is executing."""

        thread = self.connect_to_frame_base_globals()
        frame1 = thread.GetFrameAtIndex(1)
        self.assertTrue(frame1.IsValid())
        self.assertIn("main", frame1.GetFunctionName())

        # A variable relative to that frame base is only reachable by reading the
        # global, so these values are what came back from it.
        i = frame1.FindVariable("i")
        self.assertTrue(i.IsValid())
        self.assertEqual(i.GetValueAsUnsigned(), 1)

        j = frame1.FindVariable("j")
        self.assertTrue(j.IsValid())
        self.assertEqual(j.GetValueAsUnsigned(), 2)

        # The global belongs to the instance the frame is executing, and that
        # instance is what names it. The frame the read goes through does not.
        self.assertPacketLogReceived(
            [global_read_packet(OUTER_FRAME_BASE_GLOBAL_INDEX)]
        )
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_variable_located_through_global_in_innermost_frame(self):
        """Test that a variable the innermost frame locates through a Wasm global
        is read from the instance that frame is executing."""

        # The innermost frame is where a stop leaves the program, and its
        # register context is the one the thread carries, which exists before any
        # frame does. The instance that frame is executing is therefore only
        # known once the stub has been asked for its call stack.
        thread = self.connect_to_frame_base_globals()
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid())
        self.assertIn("add", frame0.GetFunctionName())

        a = frame0.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsUnsigned(), 1)

        b = frame0.FindVariable("b")
        self.assertTrue(b.IsValid())
        self.assertEqual(b.GetValueAsUnsigned(), 2)

        self.assertPacketLogReceived(
            [global_read_packet(INNER_FRAME_BASE_GLOBAL_INDEX)]
        )
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_variable_located_through_global_without_reported_call_stack(self):
        """Test that a variable a frame locates through a Wasm global is read
        from the instance that frame is executing while the stub reports no call
        stack."""

        # LLDB makes a frame from the registers the stub reports when the Wasm
        # unwinder contributes none. The program counter of that frame is the
        # only place the instance it executes is recorded, so a read that looked
        # for the instance anywhere else would be left without one and fall back
        # to naming a frame the stub never reported.
        thread = self.connect_to_frame_base_globals(call_stacks=[])
        self.assertEqual(1, thread.GetNumFrames())

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid())
        self.assertIn("add", frame0.GetFunctionName())

        a = frame0.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsUnsigned(), 1)

        self.assertPacketLogReceived(
            [global_read_packet(INNER_FRAME_BASE_GLOBAL_INDEX)]
        )
        self.assertEqual([], self.global_reads_received(instance=False))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_variable_located_through_global_legacy_server(self):
        """Test that a variable the innermost frame locates through a Wasm global
        is read through that frame when the stub cannot be told which instance to
        read."""

        thread = self.connect_to_frame_base_globals(supports_instance=False)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid())
        self.assertIn("add", frame0.GetFunctionName())

        a = frame0.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertEqual(a.GetValueAsUnsigned(), 1)

        # A frame stands in for the instance it is executing, which is all such a
        # stub can be asked, so the same global stays within reach through the
        # frame.
        self.assertPacketLogReceived(["qWasmGlobal:0;0"])
        self.assertEqual([], self.global_reads_received(instance=True))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_legacy_server(self):
        """Test that a global is read through the frame executing the instance
        that holds it when the stub cannot be told which instance to read."""

        call_stacks = [WasmCallStack([WasmStackFrame(0x019C)])]
        process, globals_addr = self.connect_to_globals(
            call_stacks, supports_instance=False
        )

        error = lldb.SBError()
        data = process.ReadMemory(globals_addr + 0, 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0x2A)

        # A global at an index the frame index cannot be mistaken for, so that
        # the frame and the global keep their places in the packet.
        data = process.ReadMemory(globals_addr + 1, 8, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), 0xDEADBEEF)

        # A stub is only asked for what it advertised, so the form it never
        # offered is not tried on it.
        self.assertPacketLogReceived(["qWasmGlobal:0;0", "qWasmGlobal:0;1"])
        self.assertEqual([], self.global_reads_received(instance=True))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_legacy_server_without_reported_call_stack(self):
        """Test that a global is out of reach of a stub that can only read a
        global through a frame while it reports no call stack."""

        process, globals_addr = self.connect_to_globals([], supports_instance=False)

        error = lldb.SBError()
        process.ReadMemory(globals_addr + 0, 4, error)
        self.assertFalse(error.Success())
        self.assertIn("can only read a global through a frame", error.GetCString())
        self.assertEqual([], self.packets_received("qWasmGlobal:"))

    @skipIfAsan
    @skipIfXmlSupportMissing
    def test_read_global_legacy_server_other_instance(self):
        """Test that a frame executing one instance is not read as a frame of
        another when the stub can only read a global through a frame."""

        module = self.build_wasm_module("test_wasm")
        second_module = self.build_wasm_module(
            "test_wasm_second",
            module_id=SECOND_MODULE_ID,
            global_values=SECOND_WASM_GLOBALS,
        )

        call_stacks = [WasmCallStack([WasmStackFrame(0x019C)])]
        target, process = self.connect_to_modules(
            [module, second_module], call_stacks, supports_instance=False
        )
        self.assertEqual(2, target.GetNumModules())

        # The only frame is executing the first instance, and a frame of one
        # instance indexes no globals of another. Reading a global of the second
        # instance fails rather than answering with a global of the first.
        error = lldb.SBError()
        process.ReadMemory(self.get_globals_address(target, second_module), 4, error)
        self.assertFalse(error.Success())
        self.assertIn("can only read a global through a frame", error.GetCString())

        # The instance the frame is executing is still within reach.
        error = lldb.SBError()
        data = process.ReadMemory(self.get_globals_address(target, module), 4, error)
        self.assertSuccess(error)
        self.assertEqual(int.from_bytes(data, "little"), WASM_GLOBALS[0][1])

        self.assertPacketLogReceived(["qWasmGlobal:0;0"])

    @skipIfXmlSupportMissing
    def test_non_wasm_process(self):
        """Test that the plugin falls back to plain GDB remote debugging when
        it is requested by name for a process that isn't WebAssembly."""

        class NonWasmResponder(MockGDBServerResponder):
            def qHostInfo(self):
                return "triple:%s;ptrsize:8;endian:little;" % hex_encode_bytes(
                    "x86_64-unknown-linux-gnu"
                )

            def qfThreadInfo(self):
                return "m1"

            def haltReason(self):
                return "T02thread:1;threads:1;thread-pcs:10001bc00;"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                          </feature>
                        </target>""",
                        False,
                    )
                return None, False

        self.server.responder = NonWasmResponder()

        target = self.dbg.CreateTarget("")
        process = self.connect(target, "wasm")
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        self.assertEqual(process.GetPluginName(), "wasm")
        self.assertIn("x86_64", target.GetTriple())

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), 0x10001BC00)
        self.assertNotIn(
            "qWasmCallStack", "".join(self.server.responder.packetLog.get_received())
        )

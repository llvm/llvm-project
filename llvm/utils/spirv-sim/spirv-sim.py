#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from instructions import *
from typing import Any, Iterable, Callable, Optional, Tuple, List, Dict
import argparse
import fileinput
import inspect
import re
import sys

RE_EXPECTS = re.compile(r"^([0-9]+,)*[0-9]+$")


# Parse the SPIR-V instructions. Some instructions are ignored because
# not required to simulate this module.
# Instructions are to be implemented in instructions.py
def parseInstruction(i):
    IGNORED = set(
        [
            "OpCapability",
            "OpMemoryModel",
            "OpExecutionMode",
            "OpExtension",
            "OpSource",
            "OpTypeInt",
            "OpTypeStruct",
            "OpTypeFloat",
            "OpTypeBool",
            "OpTypeVoid",
            "OpTypeFunction",
            "OpTypePointer",
            "OpTypeArray",
        ]
    )
    if i.opcode() in IGNORED:
        return None

    try:
        Type = getattr(sys.modules["instructions"], i.opcode())
    except AttributeError:
        raise RuntimeError(f"Unsupported instruction {i}")
    if not inspect.isclass(Type):
        raise RuntimeError(
            f"{i} instruction definition is not a class. Did you used 'def' instead of 'class'?"
        )
    return Type(i.line)


# Split a list of instructions into pieces. Pieces are delimited by instructions of the type splitType.
# The delimiter is the first instruction of the next piece.
# This function returns no empty pieces:
# - if 2 subsequent delimiters will mean 2 pieces. One with only the first delimiter, and the second
#   with the delimiter and following instructions.
# - if the first instruction is a delimiter, the first piece will begin with this delimiter.
def splitInstructions(
    splitType: type, instructions: Iterable[Instruction]
) -> List[List[Instruction]]:
    blocks: List[List[Instruction]] = [[]]
    for instruction in instructions:
        if isinstance(instruction, splitType) and len(blocks[-1]) > 0:
            blocks.append([])
        blocks[-1].append(instruction)
    return blocks


# Defines a BasicBlock in the simulator.
# Begins at an OpLabel, and ends with a control-flow instruction.
class BasicBlock:
    def __init__(self, instructions) -> None:
        assert isinstance(instructions[0], OpLabel)
        # The name of the basic block, which is the register of the leading
        # OpLabel.
        self._name = instructions[0].output_register()
        # The list of instructions belonging to this block.
        self._instructions = instructions[1:]

    # Returns the name of this basic block.
    def name(self):
        return self._name

    # Returns the instruction at index in this basic block.
    def __getitem__(self, index: int) -> Instruction:
        return self._instructions[index]

    # Returns the number of instructions in this basic block, excluding the
    # leading OpLabel.
    def __len__(self):
        return len(self._instructions)

    def dump(self):
        print(f"        {self._name}:")
        for instruction in self._instructions:
            print(f"        {instruction}")


# Defines a Function in the simulator.
class Function:
    def __init__(self, instructions) -> None:
        assert isinstance(instructions[0], OpFunction)
        # The name of the function (name of the register returned by OpFunction).
        self._name: str = instructions[0].output_register()
        # The list of basic blocks that belongs to this function.
        self._basic_blocks: List[BasicBlock] = []
        # The variables local to this function.
        self._variables: List[OpVariable] = [
            x for x in instructions if isinstance(x, OpVariable)
        ]

        assert isinstance(instructions[-1], OpFunctionEnd)
        body = filter(lambda x: not isinstance(x, OpVariable), instructions[1:-1])
        for block in splitInstructions(OpLabel, body):
            self._basic_blocks.append(BasicBlock(block))

    # Returns the name of this function.
    def name(self) -> str:
        return self._name

    # Returns the basic block at index in this function.
    def __getitem__(self, index: int) -> BasicBlock:
        return self._basic_blocks[index]

    # Returns the index of the basic block with the given name if found,
    # -1 otherwise.
    def get_bb_index(self, name) -> int:
        for i in range(len(self._basic_blocks)):
            if self._basic_blocks[i].name() == name:
                return i
        return -1

    def dump(self):
        print("      Variables:")
        for var in self._variables:
            print(f"        {var}")
        print("      Blocks:")
        for bb in self._basic_blocks:
            bb.dump()


# Represents an instruction pointer in the simulator.
@dataclass
class InstructionPointer:
    # The current function the IP points to.
    function: Function
    # The basic block index in function IP points to.
    basic_block: int
    # The instruction in basic_block IP points to.
    instruction_index: int

    def __str__(self):
        bb = self.function[self.basic_block]
        i = bb[self.instruction_index]
        return f"{bb.name()}:{self.instruction_index} in {self.function.name()} | {i}"

    def __hash__(self):
        return hash((self.function.name(), self.basic_block, self.instruction_index))

    # Returns the basic block IP points to.
    def bb(self) -> BasicBlock:
        return self.function[self.basic_block]

    # Returns the instruction IP points to.
    def instruction(self):
        return self.function[self.basic_block][self.instruction_index]

    # Increment IP by 1. This only works inside a basic-block boundary.
    # Incrementing IP when at the boundary of a basic block will fail.
    def __add__(self, value: int):
        bb = self.function[self.basic_block]
        assert len(bb) > self.instruction_index + value
        return InstructionPointer(
            self.function, self.basic_block, self.instruction_index + value
        )


# Defines a Lane in this simulator.
class Lane:
    # The registers known by this lane.
    _registers: Dict[str, Any]
    # The current IP of this lane.
    _ip: Optional[InstructionPointer]
    # If this lane running.
    _running: bool
    # The wave this lane belongs to.
    _wave: Wave
    # The callstack of this lane. Each tuple represents 1 call.
    # The first element is the IP the function will return to.
    # The second element is the callback to call to store the return value
    # into the correct register.
    _callstack: List[Tuple[InstructionPointer, Callable[[Any], None]]]

    _previous_bb: Optional[BasicBlock]
    _current_bb: Optional[BasicBlock]

    def __init__(self, wave: Wave, tid: int) -> None:
        self._registers = dict()
        self._ip = None
        self._running = True
        self._wave = wave
        self._callstack = []

        # The index of this lane in the wave.
        self._tid = tid
        # The last BB this lane was executing into.
        self._previous_bb = None
        # The current BB this lane is executing into.
        self._current_bb = None

    # Returns the lane/thread ID of this lane in its wave.
    def tid(self) -> int:
        return self._tid

    # Returns true is this lane if the first by index in the current active tangle.
    def is_first_active_lane(self) -> bool:
        return self._tid == self._wave.get_first_active_lane_index()

    # Broadcast value into the registers of all active lanes.
    def broadcast_register(self, register: str, value: Any) -> None:
        self._wave.broadcast_register(register, value)

    # Returns the IP this lane is currently at.
    def ip(self) -> InstructionPointer:
        assert self._ip is not None
        return self._ip

    # Returns true if this lane is running, false otherwise.
    # Running means not dead. An inactive lane is running.
    def running(self) -> bool:
        return self._running

    # Set the register at "name" to "value" in this lane.
    def set_register(self, name: str, value: Any) -> None:
        self._registers[name] = value

    # Get the value in register "name" in this lane.
    # If allow_undef is true, fetching an unknown register won't fail.
    def get_register(self, name: str, allow_undef: bool = False) -> Optional[Any]:
        if allow_undef and name not in self._registers:
            return None
        return self._registers[name]

    def set_ip(self, ip: InstructionPointer) -> None:
        if ip.bb() != self._current_bb:
            self._previous_bb = self._current_bb
            self._current_bb = ip.bb()
        self._ip = ip

    def get_previous_bb_name(self):
        return self._previous_bb.name()

    def handle_convergence_header(self, instruction):
        self._wave.handle_convergence_header(self, instruction)

    def do_call(self, ip, output_register):
        return_ip = None if self._ip is None else self._ip + 1
        self._callstack.append(
            (return_ip, lambda value: self.set_register(output_register, value))
        )
        self.set_ip(ip)

    def do_return(self, value):
        ip, callback = self._callstack[-1]
        self._callstack.pop()

        callback(value)
        if len(self._callstack) == 0:
            self._running = False
        else:
            self.set_ip(ip)


# Represents the SPIR-V module in the simulator.
class Module:
    _functions: Dict[str, Function]
    _prolog: List[Instruction]
    _globals: List[Instruction]
    _name2reg: Dict[str, str]
    _reg2name: Dict[str, str]

    def __init__(self, instructions) -> None:
        chunks = splitInstructions(OpFunction, instructions)

        # The instructions located outside of all functions.
        self._prolog = chunks[0]
        # The functions in this module.
        self._functions = {}
        # Global variables in this module.
        self._globals = [
            x
            for x in instructions
            if isinstance(x, OpVariable) or issubclass(type(x), OpConstant)
        ]

        # Helper dictionaries to get real names of registers, or registers by names.
        self._name2reg = {}
        self._reg2name = {}
        for instruction in instructions:
            if isinstance(instruction, OpName):
                name = instruction.name()
                reg = instruction.decoratedRegister()
                self._name2reg[name] = reg
                self._reg2name[reg] = name

        for chunk in chunks[1:]:
            function = Function(chunk)
            assert function.name() not in self._functions
            self._functions[function.name()] = function

    # Returns the register matching "name" if any, None otherwise.
    # This assumes names are unique.
    def getRegisterFromName(self, name):
        if name in self._name2reg:
            return self._name2reg[name]
        return None

    # Returns the name given to "register" if any, None otherwise.
    def getNameFromRegister(self, register):
        if register in self._reg2name:
            return self._reg2name[register]
        return None

    # Initialize the module before wave execution begins.
    # See Instruction::static_execution for more details.
    def initialize(self, lane):
        for instruction in self._globals:
            instruction.static_execution(lane)

        # Initialize builtins
        for instruction in self._prolog:
            if isinstance(instruction, OpDecorate):
                instruction.static_execution(lane)

    def execute_one_instruction(self, lane: Lane, ip: InstructionPointer) -> None:
        ip.instruction().runtime_execution(self, lane)

    # Returns the first valid IP for the function defined by the given register.
    # Calling this with a register not returned by OpFunction is illegal.
    def get_function_entry(self, register: str) -> InstructionPointer:
        if register not in self._functions:
            raise RuntimeError(f"Function defining {register} not found.")
        return InstructionPointer(self._functions[register], 0, 0)

    # Returns the first valid IP for the basic block defined by register.
    # Calling this with a register not returned by an OpLabel is illegal.
    def get_bb_entry(self, register: str) -> InstructionPointer:
        for name, function in self._functions.items():
            index = function.get_bb_index(register)
            if index != -1:
                return InstructionPointer(function, index, 0)
        raise RuntimeError(f"Instruction defining {register} not found.")

    # Returns the list of function names in this module.
    # If an OpName exists for this function, returns the pretty name, else
    # returns the register name.
    def get_function_names(self):
        return [self.getNameFromRegister(reg) for reg, func in self._functions.items()]

    # Returns the global variables defined in this module.
    def variables(self) -> Iterable:
        return [x.output_register() for x in self._globals]

    def dump(self, function_name: Optional[str] = None):
        print("Module:")
        print("  globals:")
        for instruction in self._globals:
            print(f"    {instruction}")

        if function_name is None:
            print("  functions:")
            for register, function in self._functions.items():
                name = self.getNameFromRegister(register)
                print(f"  Function {register} ({name})")
                function.dump()
            return

        register = self.getRegisterFromName(function_name)
        print(f"  function {register} ({function_name}):")
        if register is not None:
            self._functions[register].dump()
        else:
            print(f"    error: cannot find function.")


# Defines a convergence requirement for the simulation:
# A list of lanes impacted by a merge and possibly the associated
# continue target.
@dataclass
class ConvergenceRequirement:
    mergeTarget: InstructionPointer
    continueTarget: Optional[InstructionPointer]
    impactedLanes: set[int]


Task = Dict[InstructionPointer, List[Lane]]


# Defines a Lane group/Wave in the simulator.
class Wave:
    # The module this wave will execute.
    _module: Module
    # The lanes this wave will be composed of.
    _lanes: List[Lane]
    # The instructions scheduled for execution.
    _tasks: Task
    # The actual requirements to comply with when executing instructions.
    # E.g: the set of lanes required to merge before executing the merge block.
    _convergence_requirements: List[ConvergenceRequirement]
    # The indices of the active lanes for the current executing instruction.
    _active_lane_indices: set[int]

    def __init__(self, module, wave_size: int) -> None:
        assert wave_size > 0
        self._module = module
        self._lanes = []

        for i in range(wave_size):
            self._lanes.append(Lane(self, i))

        self._tasks = {}
        self._convergence_requirements = []
        # The indices of the active lanes for the current executing instruction.
        self._active_lane_indices = set()

    # Returns True if the given IP can be executed for the given list of lanes.
    def _is_task_candidate(self, ip: InstructionPointer, lanes: List[Lane]):
        merged_lanes: set[int] = set()
        for lane in self._lanes:
            if not lane.running():
                merged_lanes.add(lane.tid())

        for requirement in self._convergence_requirements:
            # This task is not executing a merge or continue target.
            # Adding all lanes at those points into the ignore list.
            if requirement.mergeTarget != ip and requirement.continueTarget != ip:
                for tid in requirement.impactedLanes:
                    if self._lanes[tid].ip() == requirement.mergeTarget:
                        merged_lanes.add(tid)
                    if self._lanes[tid].ip() == requirement.continueTarget:
                        merged_lanes.add(tid)
                continue

            # This task is executing the current requirement continue/merge
            # target.
            for tid in requirement.impactedLanes:
                lane = self._lanes[tid]
                if not lane.running():
                    continue

                if lane.tid() in merged_lanes:
                    continue

                if ip == requirement.mergeTarget:
                    if lane.ip() != requirement.mergeTarget:
                        return False
                else:
                    if (
                        lane.ip() != requirement.mergeTarget
                        and lane.ip() != requirement.continueTarget
                    ):
                        return False
        return True

    # Returns the next task we can schedule. This must always return a task.
    # Calling this when all lanes are dead is invalid.
    def _get_next_runnable_task(self) -> Tuple[InstructionPointer, List[Lane]]:
        candidate = None
        for ip, lanes in self._tasks.items():
            if len(lanes) == 0:
                continue
            if self._is_task_candidate(ip, lanes):
                candidate = ip
                break

        if candidate:
            lanes = self._tasks[candidate]
            del self._tasks[ip]
            return (candidate, lanes)
        raise RuntimeError("No task to execute. Deadlock?")

    # Handle an encountered merge instruction for the given lane.
    def handle_convergence_header(self, lane: Lane, instruction: MergeInstruction):
        mergeTarget = self._module.get_bb_entry(instruction.merge_location())
        for requirement in self._convergence_requirements:
            if requirement.mergeTarget == mergeTarget:
                requirement.impactedLanes.add(lane.tid())
                return

        continueTarget = None
        if instruction.continue_location():
            continueTarget = self._module.get_bb_entry(instruction.continue_location())
        requirement = ConvergenceRequirement(
            mergeTarget, continueTarget, set([lane.tid()])
        )
        self._convergence_requirements.append(requirement)

    # Returns true if some instructions are scheduled for execution.
    def _has_tasks(self) -> bool:
        return len(self._tasks) > 0

    # Returns the index of the first active lane right now.
    def get_first_active_lane_index(self) -> int:
        return min(self._active_lane_indices)

    # Broadcast the given value to all active lane registers.
    def broadcast_register(self, register: str, value: Any) -> None:
        for tid in self._active_lane_indices:
            self._lanes[tid].set_register(register, value)

    # Returns the entrypoint of the function associated with 'name'.
    # Calling this function with an invalid name is illegal.
    def _get_function_entry_from_name(self, name: str) -> InstructionPointer:
        register = self._module.getRegisterFromName(name)
        assert register is not None
        return self._module.get_function_entry(register)

    # Run the wave on the function 'function_name' until all lanes are dead.
    # If verbose is True, execution trace is printed.
    # Returns the value returned by the function for each lane.
    def run(self, function_name: str, verbose: bool = False) -> List[Any]:
        for t in self._lanes:
            self._module.initialize(t)

        entry_ip = self._get_function_entry_from_name(function_name)
        assert entry_ip is not None
        for t in self._lanes:
            t.do_call(entry_ip, "__shader_output__")

        self._tasks[self._lanes[0].ip()] = self._lanes
        while self._has_tasks():
            ip, lanes = self._get_next_runnable_task()
            self._active_lane_indices = set([x.tid() for x in lanes])
            if verbose:
                print(
                    f"Executing with lanes {self._active_lane_indices}: {ip.instruction()}"
                )

            for lane in lanes:
                self._module.execute_one_instruction(lane, ip)
                if not lane.running():
                    continue

                if lane.ip() in self._tasks:
                    self._tasks[lane.ip()].append(lane)
                else:
                    self._tasks[lane.ip()] = [lane]

            if verbose and ip.instruction().has_output_register():
                register = ip.instruction().output_register()
                print(
                    f"   {register:3} = {[ x.get_register(register, allow_undef=True) for x in lanes ]}"
                )

        output = []
        for lane in self._lanes:
            output.append(lane.get_register("__shader_output__"))
        return output

    def dump_register(self, register: str) -> None:
        for lane in self._lanes:
            print(
                f" Lane {lane.tid():2} | {register:3} = {lane.get_register(register)}"
            )


parser = argparse.ArgumentParser(
    description="simulator", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-i", "--input", help="Text SPIR-V to read from", required=False, default="-"
)
parser.add_argument("-f", "--function", help="Function to execute")
parser.add_argument("-w", "--wave", help="Wave size", default=32, required=False)
parser.add_argument(
    "-e",
    "--expects",
    help="Expected results per lanes, expects a list of values. Ex: '1, 2, 3'.",
)
parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
args = parser.parse_args()


def load_instructions(filename: str):
    if filename is None:
        return []

    if filename.strip() != "-":
        try:
            with open(filename, "r") as f:
                lines = f.read().split("\n")
        except Exception:  # (FileNotFoundError, PermissionError):
            return []
    else:
        lines = sys.stdin.readlines()

    # Remove leading/trailing whitespaces.
    lines = [x.strip() for x in lines]
    # Strip comments.
    lines = [x for x in filter(lambda x: len(x) != 0 and x[0] != ";", lines)]

    instructions = []
    for i in [Instruction(x) for x in lines]:
        out = parseInstruction(i)
        if out != None:
            instructions.append(out)
    return instructions


def main():
    if args.expects is None or not RE_EXPECTS.match(args.expects):
        print("Invalid format for --expects/-e flag.", file=sys.stderr)
        sys.exit(1)
    if args.function is None:
        print("Invalid format for --function/-f flag.", file=sys.stderr)
        sys.exit(1)
    try:
        int(args.wave)
    except ValueError:
        print("Invalid format for --wave/-w flag.", file=sys.stderr)
        sys.exit(1)

    expected_results = [int(x.strip()) for x in args.expects.split(",")]
    wave_size = int(args.wave)
    if len(expected_results) != wave_size:
        print("Wave size != expected result array size", file=sys.stderr)
        sys.exit(1)

    instructions = load_instructions(args.input)
    if len(instructions) == 0:
        print("Invalid input. Expected a text SPIR-V module.")
        sys.exit(1)

    module = Module(instructions)
    if args.verbose:
        module.dump()
        module.dump(args.function)

    function_names = module.get_function_names()
    if args.function not in function_names:
        print(
            f"'{args.function}' function not found. Known functions are:",
            file=sys.stderr,
        )
        for name in function_names:
            print(f" - {name}", file=sys.stderr)
        sys.exit(1)

    wave = Wave(module, wave_size)
    results = wave.run(args.function, verbose=args.verbose)

    if expected_results != results:
        print("Expected != Observed", file=sys.stderr)
        print(f"{expected_results} != {results}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


main()

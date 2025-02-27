from typing import Optional, List


# Base class for an instruction. To implement a basic instruction that doesn't
# impact the control-flow, create a new class inheriting from this.
class Instruction:
    # Contains the name of the output register, if any.
    _result: Optional[str]
    # Contains the instruction opcode.
    _opcode: str
    # Contains all the instruction operands, except result and opcode.
    _operands: List[str]

    def __init__(self, line: str):
        self.line = line
        tokens = line.split()
        if len(tokens) > 1 and tokens[1] == "=":
            self._result = tokens[0]
            self._opcode = tokens[2]
            self._operands = tokens[3:] if len(tokens) > 2 else []
        else:
            self._result = None
            self._opcode = tokens[0]
            self._operands = tokens[1:] if len(tokens) > 1 else []

    def __str__(self):
        if self._result is None:
            return f"      {self._opcode} {self._operands}"
        return f"{self._result:3} = {self._opcode} {self._operands}"

    # Returns the instruction opcode.
    def opcode(self) -> str:
        return self._opcode

    # Returns the instruction operands.
    def operands(self) -> List[str]:
        return self._operands

    # Returns the instruction output register. Calling this function is
    # only allowed if has_output_register() is true.
    def output_register(self) -> str:
        assert self._result is not None
        return self._result

    # Returns true if this function has an output register. False otherwise.
    def has_output_register(self) -> bool:
        return self._result is not None

    # This function is used to initialize state related to this instruction
    # before module execution begins. For example, global Input variables
    # can use this to store the lane ID into the register.
    def static_execution(self, lane):
        pass

    # This function is called everytime this instruction is executed by a
    # tangle. This function should not be directly overriden, instead see
    # _impl and _advance_ip.
    def runtime_execution(self, module, lane):
        self._impl(module, lane)
        self._advance_ip(module, lane)

    # This function needs to be overriden if your instruction can be executed.
    # It implements the logic of the instruction.
    # 'Static' instructions like OpConstant should not override this since
    # they are not supposed to be executed at runtime.
    def _impl(self, module, lane):
        raise RuntimeError(f"Unimplemented instruction {self}")

    # By default, IP is incremented to point to the next instruction.
    # If the instruction modifies IP (like OpBranch), this must be overridden.
    def _advance_ip(self, module, lane):
        lane.set_ip(lane.ip() + 1)


# Those are parsed, but never executed.
class OpEntryPoint(Instruction):
    pass


class OpFunction(Instruction):
    pass


class OpFunctionEnd(Instruction):
    pass


class OpLabel(Instruction):
    pass


class OpVariable(Instruction):
    pass


class OpName(Instruction):
    def name(self) -> str:
        return self._operands[1][1:-1]

    def decoratedRegister(self) -> str:
        return self._operands[0]


# The only decoration we use if the BuiltIn one to initialize the values.
class OpDecorate(Instruction):
    def static_execution(self, lane):
        if self._operands[1] == "LinkageAttributes":
            return

        assert (
            self._operands[1] == "BuiltIn"
            and self._operands[2] == "SubgroupLocalInvocationId"
        )
        lane.set_register(self._operands[0], lane.tid())


# Constants
class OpConstant(Instruction):
    def static_execution(self, lane):
        lane.set_register(self._result, int(self._operands[1]))


class OpConstantTrue(OpConstant):
    def static_execution(self, lane):
        lane.set_register(self._result, True)


class OpConstantFalse(OpConstant):
    def static_execution(self, lane):
        lane.set_register(self._result, False)


class OpConstantComposite(OpConstant):
    def static_execution(self, lane):
        result = []
        for op in self._operands[1:]:
            result.append(lane.get_register(op))
        lane.set_register(self._result, result)


# Control flow instructions
class OpFunctionCall(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        entry = module.get_function_entry(self._operands[1])
        lane.do_call(entry, self._result)


class OpReturn(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        lane.do_return(None)


class OpReturnValue(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        lane.do_return(lane.get_register(self._operands[0]))


class OpBranch(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        lane.set_ip(module.get_bb_entry(self._operands[0]))
        pass


class OpBranchConditional(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        condition = lane.get_register(self._operands[0])
        if condition:
            lane.set_ip(module.get_bb_entry(self._operands[1]))
        else:
            lane.set_ip(module.get_bb_entry(self._operands[2]))


class OpSwitch(Instruction):
    def _impl(self, module, lane):
        pass

    def _advance_ip(self, module, lane):
        value = lane.get_register(self._operands[0])
        default_label = self._operands[1]
        i = 2
        while i < len(self._operands):
            imm = int(self._operands[i])
            label = self._operands[i + 1]
            if value == imm:
                lane.set_ip(module.get_bb_entry(label))
                return
            i += 2
        lane.set_ip(module.get_bb_entry(default_label))


class OpUnreachable(Instruction):
    def _impl(self, module, lane):
        raise RuntimeError("This instruction should never be executed.")


# Convergence instructions
class MergeInstruction(Instruction):
    def merge_location(self):
        return self._operands[0]

    def continue_location(self):
        return None if len(self._operands) < 3 else self._operands[1]

    def _impl(self, module, lane):
        lane.handle_convergence_header(self)


class OpLoopMerge(MergeInstruction):
    pass


class OpSelectionMerge(MergeInstruction):
    pass


# Other instructions
class OpBitcast(Instruction):
    def _impl(self, module, lane):
        # TODO: find out the type from the defining instruction.
        # This can only work for DXC.
        if self._operands[0] == "%int":
            lane.set_register(self._result, int(lane.get_register(self._operands[1])))
        else:
            raise RuntimeError("Unsupported OpBitcast operand")


class OpAccessChain(Instruction):
    def _impl(self, module, lane):
        # Python dynamic types allows me to simplify. As long as the SPIR-V
        # is legal, this should be fine.
        # Note: SPIR-V structs are stored as tuples
        value = lane.get_register(self._operands[1])
        for operand in self._operands[2:]:
            value = value[lane.get_register(operand)]
        lane.set_register(self._result, value)


class OpCompositeConstruct(Instruction):
    def _impl(self, module, lane):
        output = []
        for op in self._operands[1:]:
            output.append(lane.get_register(op))
        lane.set_register(self._result, output)


class OpCompositeExtract(Instruction):
    def _impl(self, module, lane):
        value = lane.get_register(self._operands[1])
        output = value
        for op in self._operands[2:]:
            output = output[int(op)]
        lane.set_register(self._result, output)


class OpStore(Instruction):
    def _impl(self, module, lane):
        lane.set_register(self._operands[0], lane.get_register(self._operands[1]))


class OpLoad(Instruction):
    def _impl(self, module, lane):
        lane.set_register(self._result, lane.get_register(self._operands[1]))


class OpIAdd(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS + RHS)


class OpISub(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS - RHS)


class OpIMul(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS * RHS)


class OpLogicalNot(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        lane.set_register(self._result, not LHS)


class _LessThan(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS < RHS)


class _GreaterThan(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS > RHS)


class OpSLessThan(_LessThan):
    pass


class OpULessThan(_LessThan):
    pass


class OpSGreaterThan(_GreaterThan):
    pass


class OpUGreaterThan(_GreaterThan):
    pass


class OpIEqual(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS == RHS)


class OpINotEqual(Instruction):
    def _impl(self, module, lane):
        LHS = lane.get_register(self._operands[1])
        RHS = lane.get_register(self._operands[2])
        lane.set_register(self._result, LHS != RHS)


class OpPhi(Instruction):
    def _impl(self, module, lane):
        previousBBName = lane.get_previous_bb_name()
        i = 1
        while i < len(self._operands):
            label = self._operands[i + 1]
            if label == previousBBName:
                lane.set_register(self._result, lane.get_register(self._operands[i]))
                return
            i += 2
        raise RuntimeError("previousBB not in the OpPhi _operands")


class OpSelect(Instruction):
    def _impl(self, module, lane):
        condition = lane.get_register(self._operands[1])
        value = lane.get_register(self._operands[2 if condition else 3])
        lane.set_register(self._result, value)


# Wave intrinsics
class OpGroupNonUniformBroadcastFirst(Instruction):
    def _impl(self, module, lane):
        assert lane.get_register(self._operands[1]) == 3
        if lane.is_first_active_lane():
            lane.broadcast_register(self._result, lane.get_register(self._operands[2]))


class OpGroupNonUniformElect(Instruction):
    def _impl(self, module, lane):
        lane.set_register(self._result, lane.is_first_active_lane())

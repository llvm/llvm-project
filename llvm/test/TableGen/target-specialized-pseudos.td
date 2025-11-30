// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s -DONECASE -o - | FileCheck -check-prefixes=CHECK,ONECASE %s
// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s -DALLCASES -o - | FileCheck -check-prefixes=CHECK,ALLCASES %s
// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include %s -DERROR -o /dev/null 2>&1 | FileCheck -check-prefix=ERROR %s

// CHECK: namespace llvm::MyTarget {
// CHECK: enum {
// CHECK: LOAD_STACK_GUARD = [[LOAD_STACK_GUARD_OPCODE:[0-9]+]],
// CHECK: PREALLOCATED_ARG = [[PREALLOCATED_ARG_OPCODE:[0-9]+]],
// CHECK: PATCHABLE_EVENT_CALL = [[PATCHABLE_EVENT_CALL_OPCODE:[0-9]+]],
// CHECK: PATCHABLE_TYPED_EVENT_CALL = [[PATCHABLE_TYPED_EVENT_CALL_OPCODE:[0-9]+]],

// Make sure no enum entry is emitted for MY_LOAD_STACK_GUARD
// CHECK: G_UBFX = [[G_UBFX_OPCODE:[0-9]+]],
// CHECK-NEXT: MY_MOV = [[MY_MOV_OPCODE:[0-9]+]],
// CHECK-NEXT: INSTRUCTION_LIST_END = [[INSTR_LIST_END_OPCODE:[0-9]+]]


// CHECK: extern const MyTargetInstrTable MyTargetDescs = {
// CHECK-NEXT: {
// CHECK-NEXT: { [[MY_MOV_OPCODE]],	2,	1,	2,	0,	0,	0,	{{[0-9]+}},	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // MY_MOV
// CHECK-NEXT: { [[G_UBFX_OPCODE]],	4,	1,	0,	0,	0,	0,	{{[0-9]+}},	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::PreISelOpcode)|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // G_UBFX

// ONECASE: { [[LOAD_STACK_GUARD_OPCODE]],	1,	1,	0,	0,	0,	0,	[[LOAD_STACK_GUARD_OP_ENTRY:[0-9]+]],	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Rematerializable)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // MY_LOAD_STACK_GUARD

// ALLCASES: { [[PATCHABLE_TYPED_EVENT_CALL_OPCODE]],	3,	0,	0,	0,	0,	0,	[[PATCHABLE_TYPED_EVENT_CALL_OP_ENTRY:[0-9]+]],	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // anonymous_
// ALLCASES: { [[PATCHABLE_EVENT_CALL_OPCODE]],	2,	0,	0,	0,	0,	0,	[[PATCHABLE_EVENT_CALL_OP_ENTRY:[0-9]+]],	MyTargetImpOpBase + 0, 0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::Call)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::MayStore)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // anonymous_
// ALLCASES: { [[PREALLOCATED_ARG_OPCODE]],	3,	1,	0,	0,	0,	0,	[[PREALLOCATED_ARG_OP_ENTRY:[0-9]+]],	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::UsesCustomInserter)|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // anonymous_
// ALLCASES: { [[LOAD_STACK_GUARD_OPCODE]],	1,	1,	0,	0,	0,	0,	[[LOAD_STACK_GUARD_OP_ENTRY:[0-9]+]],	MyTargetImpOpBase + 0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Rematerializable)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // anonymous_

// CHECK: /* 0 */ { -1, 0, MCOI::OPERAND_UNKNOWN, 0 },

// ONECASE: /* [[LOAD_STACK_GUARD_OP_ENTRY]] */ { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },

// ALLCASES: /* [[LOAD_STACK_GUARD_OP_ENTRY]] */ { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// ALLCASES: /* [[PREALLOCATED_ARG_OP_ENTRY]] */ { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 },
// ALLCASES: /* [[PATCHABLE_EVENT_CALL_OP_ENTRY]] */ { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 },
// ALLCASES: /* [[PATCHABLE_TYPED_EVENT_CALL_OP_ENTRY]] */ { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 },


// CHECK: const char MyTargetInstrNameData[] = {
// CHECK: /* {{[0-9]+}} */ "LOAD_STACK_GUARD\000"

include "llvm/Target/Target.td"

class MyReg<string n>
  : Register<n> {
  let Namespace = "MyTarget";
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def XRegs : RegisterClass<"MyTarget", [i64], 64, (add X0, X1)>;


class TestInstruction : Instruction {
  let Size = 2;
  let Namespace = "MyTarget";
  let hasSideEffects = false;
}

#ifdef ONECASE

// Example setting the pointer register class manually
def MY_LOAD_STACK_GUARD :
  TargetSpecializedStandardPseudoInstruction<LOAD_STACK_GUARD> {
  let Namespace = "MyTarget";
  let OutOperandList = (outs XRegs:$dst);
}

#endif

#ifdef ALLCASES

defm my_remaps : RemapAllTargetPseudoPointerOperands<XRegs>;

#endif


#ifdef ERROR

def MY_LOAD_STACK_GUARD_0 : TargetSpecializedStandardPseudoInstruction<LOAD_STACK_GUARD>;

// ERROR: :[[@LINE+1]]:5: error: multiple overrides of 'LOAD_STACK_GUARD' defined
def MY_LOAD_STACK_GUARD_1 : TargetSpecializedStandardPseudoInstruction<LOAD_STACK_GUARD>;

#endif

def MY_MOV : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "my_mov $dst, $src";
}


def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

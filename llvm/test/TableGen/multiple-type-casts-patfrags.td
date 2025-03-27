// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s | FileCheck -check-prefix=SDAG %s
// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def REG : Register<"REG">;
def GPR : RegisterClass<"MyTarget", [i16, i32], 32, (add REG)>;

def int_foo : Intrinsic<[llvm_anyint_ty, llvm_anyint_ty], []>;

def INSTR_FOO_I16_I32 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}
def INSTR_FOO_I32_I16 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}

// SDAG: 7*/ OPC_SwitchType {{.*}}, 10, /*MVT::i16*/6
// SDAG: OPC_CheckTypeRes, 1, /*MVT::i32*/7
// SDAG: OPC_MorphNodeTo2Chain, TARGET_VAL(::INSTR_FOO_I16_I32)

// GISEL: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s16
// GISEL: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32
// GISEL: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(::INSTR_FOO_I16_I32)
def : Pat<([i16, i32] (int_foo)), ([i16, i32] (INSTR_FOO_I16_I32))>;

// SDAG: 20*/ /*SwitchType*/ {{.*}} /*MVT::i32*/7
// SDAG: OPC_CheckTypeRes, 1, /*MVT::i16*/6
// SDAG: OPC_MorphNodeTo2Chain, TARGET_VAL(::INSTR_FOO_I32_I16)

// GISEL: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32
// GISEL: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s16
// GISEL: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(::INSTR_FOO_I32_I16)
def : Pat<([i32, i16] (int_foo)), ([i32, i16] (INSTR_FOO_I32_I16))>;

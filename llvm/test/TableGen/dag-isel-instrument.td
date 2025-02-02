// RUN: llvm-tblgen -gen-dag-isel -instrument-coverage -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG : Register<"REG">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add REG)>;

// CHECK-LABEL: OPC_CheckOpcode, TARGET_VAL(ISD::UDIVREM)
// CHECK: OPC_EmitNode2None, TARGET_VAL(::INSTR)
// CHECK-NEXT: Results = #2 #3
// CHECK-NEXT: OPC_Coverage, COVERAGE_IDX_VAL(0),
// CHECK-NEXT: OPC_CompleteMatch, 2, 3, 2
def INSTR : Instruction {
  let OutOperandList = (outs GPR:$r1, GPR:$r0);
  let InOperandList = (ins GPR:$t0, GPR:$t1);
  let Pattern = [(set i32:$r0, i32:$r1, (udivrem i32:$t0, i32:$t1))];
}


// CHECK: getPatternForIndex(unsigned Index)
// CHECK: static const char *PATTERN_MATCH_TABLE[]
// CHECK: return StringRef(PATTERN_MATCH_TABLE[Index]);

// CHECK: getIncludePathForIndex(unsigned Index)
// CHECK: static const char *INCLUDE_PATH_TABLE[]
// CHECK: return StringRef(INCLUDE_PATH_TABLE[Index]);

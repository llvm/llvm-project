// RUN: llvm-tblgen --gen-dag-isel -I %p/../../include %s -DIGNORE | FileCheck %s
// RUN: llvm-tblgen --gen-dag-isel -I %p/../../include %s --match-prefix=CHECK_LOOK | FileCheck --check-prefix=CHECK_LOOK %s

include "llvm/Target/Target.td"
def ISA : InstrInfo;
def Tgt : Target { let InstructionSet = ISA; }
def R0 : Register<"r0"> { let Namespace = "Tgt"; }
def RC : RegisterClass<"Tgt", [i32], 32, (add R0)>;
def Op : RegisterOperand<RC>;

class I<dag OOps, dag IOps, list<dag> Pat> : Instruction {
  let Namespace = "Tgt";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

#ifdef IGNORE
let DAGISelShouldIgnore = 1 in
def ADD : I<(outs Op:$rd), (ins Op:$rs, Op:$rt), []>;
// CHECK: static const unsigned char MatcherTable[] = {
// CHECK-NEXT:  0
// CHECK-NEXT: }; // Total Array size is 1 bytes
#else
def ADD : I<(outs Op:$rd), (ins Op:$rs, Op:$rt),
           [(set Op:$rd, (add Op:$rs, Op:$rt))]>;
// CHECK_LOOK:static const unsigned char MatcherTable[] = {
// CHECK_LOOK-NEXT:/*     0*/ OPC_CheckOpcode, TARGET_VAL(ISD::ADD),
// CHECK_LOOK-NEXT:/*     3*/ OPC_RecordChild0, // #0 = $rs
// CHECK_LOOK-NEXT:/*     4*/ OPC_RecordChild1, // #1 = $rt
// CHECK_LOOK-NEXT:/*     5*/ OPC_MorphNodeTo1None, TARGET_VAL(Tgt::ADD),
// CHECK_LOOK-NEXT:             MVT::i32, 2/*#Ops*/, 0, 1,
// CHECK_LOOK-NEXT:// Src: (add:{ *:[i32] } Op:{ *:[i32] }:$rs, Op:{ *:[i32] }:$rt) - Complexity = {{[0-9]+}}
// CHECK_LOOK-NEXT:// Dst: (ADD:{ *:[i32] } Op:{ *:[i32] }:$rs, Op:{ *:[i32] }:$rt)
// CHECK_LOOK-NEXT: 0
// CHECK_LOOK-NEXT:}; // Total Array size is 13 bytes
#endif

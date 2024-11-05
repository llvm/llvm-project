// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o /dev/null 2>&1 < %s | FileCheck %s --implicit-check-not="Skipped pattern"

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let Uses = [B0], Defs = [B0] in
def tst1 : I<(outs), (ins), [(set B0, (add B0, 1))]>;

// CHECK: Skipped pattern: unhandled discarded def
let Uses = [B0] in
def tst2 : I<(outs), (ins), [(set B0, (add B0, 1))]>;

// test courtesy @s-barannikov
def SDTBinOpWithFlagsOut : SDTypeProfile<2, 2, [
  SDTCisInt<0>,       // result
  SDTCisVT<1, i32>,   // out flags
  SDTCisSameAs<2, 0>, // lhs
  SDTCisSameAs<3, 0>  // rhs
]>;

def my_sub : SDNode<"MyTargetISD::SUB", SDTBinOpWithFlagsOut>;
def my_ineg : PatFrag<(ops node:$val), (my_sub 0, node:$val)>;

let Defs = [R1], Constraints = "$rd = $rs2" in
def tst3 : I<(outs GPR32:$rd), (ins GPR32:$rs2), []>;

// CHECK: Skipped pattern: Src pattern result has more defs than dst MI (2 def(s) vs 1 def(s))
def : Pat<(my_ineg i32:$val), (tst3 i32:$val)>;

def G_MY_SUB : GenericInstruction {
  let Namespace = "MyTarget";
  let OutOperandList = (outs type0:$dst, type1:$flags_out);
  let InOperandList = (ins type0:$lhs, type0:$rhs);
}

def : GINodeEquiv<G_MY_SUB, my_sub>;

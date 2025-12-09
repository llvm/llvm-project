// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen --gen-asm-matcher -I %p/../../include -I %t -I %S \
// RUN:   %t/inst-alias-bad-reg.td -o /dev/null 2>&1 | FileCheck %t/inst-alias-bad-reg.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-asm-matcher -I %p/../../include -I %t -I %S \
// RUN:   %t/inst-alias-static-predicates.td -o /dev/null 2>&1 | FileCheck %t/inst-alias-static-predicates.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %t -I %S \
// RUN:   %t/compress-regclass-by-hwmode.td -o /dev/null 2>&1 | FileCheck %t/compress-regclass-by-hwmode.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %t -I %S \
// RUN:   %t/compress-regclass-by-hwmode-2.td -o /dev/null 2>&1 | FileCheck %t/compress-regclass-by-hwmode-2.td --implicit-check-not="error:"

//--- Common.td
include "Common/RegClassByHwModeCommon.td"

def IsPtr64 : Predicate<"Subtarget->isPtr64()">;
def IsPtr32 : Predicate<"!Subtarget->isPtr64()">;
defvar Ptr32 = DefaultMode;
def Ptr64 : HwMode<[IsPtr64]>;
def PtrRC : RegClassByHwMode<[Ptr32, Ptr64], [XRegs, YRegs]>;

def PTR_MOV : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov $dst, $src";
  let opcode = 0;
}


//--- inst-alias-bad-reg.td
include "Common.td"
/// This should fail since X0 is not necessarily part of PtrRC.
def BAD_REG : InstAlias<"ptr_zero $rd", (PTR_MOV PtrRC:$dst, X0)>;
// CHECK: [[#@LINE-1]]:5: error: cannot resolve HwMode for PtrRC
// CHECK: Common.td:7:5: note: PtrRC defined here
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- inst-alias-static-predicates.td
include "Common.td"
/// In theory we could allow the following code since the predicates statically
/// resolve to the correct register class, but since this is non-trivial, check
// that we get a sensible-ish error instead.
let Predicates = [IsPtr32] in
def MOV_X0 : InstAlias<"mov_x0 $dst", (PTR_MOV PtrRC:$dst, X0)>;
// CHECK: [[#@LINE-1]]:5: error: cannot resolve HwMode for PtrRC
// CHECK: Common.td:7:5: note: PtrRC defined here
let Predicates = [IsPtr64] in
def MOV_Y0 : InstAlias<"mov_y0 $dst", (PTR_MOV PtrRC:$dst, Y0)>;

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

//--- compress-regclass-by-hwmode.td
include "Common.td"
def PTR_ZERO_SMALL : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins);
  let AsmString = "ptr_zero $dst";
  let opcode = 1;
  let Size = 1;
}
/// This should fail since X0 is not necessarily part of PtrRC.
def : CompressPat<(PTR_MOV PtrRC:$dst, X0),
                  (PTR_ZERO_SMALL PtrRC:$dst)>;
// CHECK: [[#@LINE-2]]:1: error: cannot resolve HwMode for PtrRC
// CHECK: Common.td:7:5: note: PtrRC defined here
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- compress-regclass-by-hwmode-2.td
include "Common.td"
def X_MOV_BIG : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "x_mov $dst, $src";
  let opcode = 1;
  let Size = 4;
}
/// This should fail since PtrRC is not necessarily part of XRegs.
/// In theory, this could be resolved depending on the Predicates but
/// for not we should just always emit an error.
let Predicates = [IsPtr32] in
def : CompressPat<(X_MOV_BIG XRegs:$dst, XRegs:$src),
                  (PTR_MOV PtrRC:$dst, PtrRC:$src)>;
// CHECK: [[#@LINE-2]]:1: error: Type mismatch between Input and Output Dag operand 'dst'
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

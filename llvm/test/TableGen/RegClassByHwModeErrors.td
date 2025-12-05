// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen --gen-asm-matcher -I %p/../../include -I %t -I %S \
// RUN:   %t/inst-alias-bad-reg.td -o /dev/null 2>&1 | FileCheck %t/inst-alias-bad-reg.td
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %t -I %S \
// RUN:   %t/compress-regclass-by-hwmode.td -o /dev/null 2>&1 | FileCheck %t/compress-regclass-by-hwmode.td

//--- Common.td
include "Common/RegClassByHwModeCommon.td"

def IsPtr64 : Predicate<"Subtarget->isPtr64()">;
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
// This should fail since X0 is not necessarily part of PtrRC.
def BAD_REG : InstAlias<"ptr_zero $rd", (PTR_MOV PtrRC:$dst, X0)>;
// CHECK: [[#@LINE-1]]:5: error: cannot resolve HwMode for PtrRC
// CHECK: Common.td:6:5: note: PtrRC defined here
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
// This should fail since X0 is not necessarily part of PtrRC.
def : CompressPat<(PTR_MOV PtrRC:$dst, X0),
                  (PTR_ZERO_SMALL PtrRC:$dst)>;
// CHECK: [[#@LINE-2]]:1: error: cannot resolve HwMode for PtrRC
// CHECK: Common.td:6:5: note: PtrRC defined here
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

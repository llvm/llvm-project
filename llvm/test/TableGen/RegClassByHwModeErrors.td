// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen --gen-asm-matcher -I %p/../../include -I %t -I %S \
// RUN:   %t/inst-alias-bad-reg.td -o /dev/null 2>&1 | FileCheck %t/inst-alias-bad-reg.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-asm-matcher -I %p/../../include -I %t -I %S \
// RUN:   %t/inst-alias-static-predicates.td -o /dev/null 2>&1 | FileCheck %t/inst-alias-static-predicates.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %t -I %S \
// RUN:   %t/compress-regclass-by-hwmode.td -o /dev/null 2>&1 | FileCheck %t/compress-regclass-by-hwmode.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %t -I %S \
// RUN:   %t/compress-regclass-by-hwmode-2.td -o /dev/null 2>&1 | FileCheck %t/compress-regclass-by-hwmode-2.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-dag-isel  -I %p/../../include -I %t -I %S \
// RUN:   %t/vt-by-hwmode-missing.td -o /dev/null 2>&1 | FileCheck %t/vt-by-hwmode-missing.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-dag-isel -I %p/../../include -I %t -I %S \
// RUN:   %t/multiple-entries-for-same-mode.td -o /dev/null 2>&1 | FileCheck %t/multiple-entries-for-same-mode.td --implicit-check-not="error:"

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
// CHECK: [[#@LINE-2]]:1: error: Error in Dag '(PTR_MOV PtrRC:$dst, X0)': Register 'X0' is not in register class 'PtrRC'
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


//--- vt-by-hwmode-missing.td
include "Common.td"
/// This should fail since we are missing a DefaultMode entry for the
/// ValueTypeByHwMode and can't resolve it for the Ptr32 (default) case.
def BadVT : ValueTypeByHwMode<[Ptr64], [i64]>;
def BadVTRegClass : RegisterClass<"MyTarget", [i64, BadVT], 64, (add Y0, Y1)>;
/// NOTE: this error only occurs for RegClassByHwMode, normal RegisterClass
/// logic for VT resolution takes a different code path and does not error.
def TEST_OK : TestInstruction {
  let OutOperandList = (outs BadVTRegClass:$dst);
  let InOperandList = (ins BadVTRegClass:$src1, BadVTRegClass:$src2);
  let AsmString = "test $dst, $src1, $src2";
  let Pattern = [(set BadVTRegClass:$dst, (add BadVTRegClass:$src1, BadVTRegClass:$src2))];
}
/// Once we use RegClassByHwMode, we get an error about missing modes:
def BadPtrRC : RegClassByHwMode<[Ptr32, Ptr64], [BadVTRegClass, YRegs]>;
// CHECK: vt-by-hwmode-missing.td:[[#@LINE-1]]:5: error: Could not resolve VT for Mode DefaultMode
// CHECK: vt-by-hwmode-missing.td:4:5: note: ValueTypeByHwMode BadVT defined here
// CHECK: vt-by-hwmode-missing.td:[[#@LINE+1]]:5: note: pattern instantiated here
def TEST : TestInstruction {
  let OutOperandList = (outs BadPtrRC:$dst);
  let InOperandList = (ins BadPtrRC:$src1, BadPtrRC:$src2);
  let AsmString = "test $dst, $src1, $src2";
  let opcode = 0;
  let Pattern = [(set BadPtrRC:$dst, (add BadPtrRC:$src1, BadPtrRC:$src2))];
}

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- multiple-entries-for-same-mode.td
include "Common.td"
/// We should get an error if the same mode is listed more than once
defvar Ptr64Alias = Ptr64;
def BadRegClass : RegClassByHwMode<[Ptr32, Ptr64, Ptr64Alias], [XRegs, YRegs, YRegs]>;
// CHECK: [[#@LINE-1]]:5: error: duplicate RegisterClass entry for HwMode Ptr64: YRegs
// Need at least one CompressPat use of the bad reg class to trigger the error:
def USE_BAD_REG_CLASS : TestInstruction {
  let OutOperandList = (outs BadRegClass:$dst);
  let InOperandList = (ins BadRegClass:$src1, BadRegClass:$src2);
  let AsmString = "bad $dst";
  let Pattern = [(set BadRegClass:$dst, (add BadRegClass:$src1, BadRegClass:$src2))];
}
def MyTargetISA : InstrInfo;
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
}

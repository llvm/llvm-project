// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

// Verify VarLenCodeEmitterGen using EncodingInfos with different HwModes.

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i64], 0, (add Reg)>;

def GR64 : RegisterOperand<RegClass>;

def HasA : Predicate<"Subtarget->hasA()">;
def HasB : Predicate<"Subtarget->hasB()">;

def ModeA : HwMode<"+a", [HasA]>;
def ModeB : HwMode<"+b", [HasB]>;

def fooTypeEncA : InstructionEncoding {
  dag Inst = (descend
    (operand "$src", 4),
    (operand "$dst", 4),
    0b00000001
  );
}

def fooTypeEncB : InstructionEncoding {
  dag Inst = (descend
    (operand "$dst", 4),
    (operand "$src", 4),
    0b00000010
  );
}

def fooTypeEncC : InstructionEncoding {
  dag Inst = (descend
    (operand "$dst", 4),
    (operand "$src", 4),
    0b00000100
  );
}

class VarLenInst : Instruction {
  let AsmString = "foo $src, $dst";
  let OutOperandList = (outs GR64:$dst);
  let InOperandList  = (ins GR64:$src);
}

// Defined in both HwModes
def foo : VarLenInst {
  let EncodingInfos = EncodingByHwMode<
    [ModeA, ModeB],
    [fooTypeEncA, fooTypeEncB]
  >;
}

// Same encoding in any HwMode
def bar : VarLenInst {
  dag Inst = (descend
    (operand "$dst", 4),
    (operand "$src", 4),
    0b00000011
  );
}

// Only defined in HwMode B.
def baz : VarLenInst {
  let EncodingInfos = EncodingByHwMode<
    [ModeB],
    [fooTypeEncC]
  >;
}

// CHECK:     static const uint64_t InstBits_ModeA[] = {
// CHECK:       UINT64_C(3),        // bar
// CHECK:       UINT64_C(1),        // foo

// CHECK:     static const uint64_t InstBits_ModeB[] = {
// CHECK:       UINT64_C(3),        // bar
// CHECK:       UINT64_C(4),        // baz
// CHECK:       UINT64_C(2),        // foo

// CHECK:     auto getInstBits_ModeA =
// CHECK:       Idx = Index_ModeA

// CHECK:     auto getInstBits_ModeB =
// CHECK:       Idx = Index_ModeB

// CHECK:     case ::bar: {
// CHECK-NOT:   switch (Mode) {
// CHECK:       Inst = getInstBits_ModeA

// CHECK:     case ::foo: {
// CHECK:       switch (Mode) {
// CHECK:       case 1: {
// CHECK:       Inst = getInstBits_ModeA
// CHECK:       case 2: {
// CHECK:       Inst = getInstBits_ModeB

// CHECK:     case ::baz: {
// CHECK:       case 1: {
// CHECK:       llvm_unreachable("Undefined encoding in this mode");
// CHECK:       case 2: {
// CHECK:       Inst = getInstBits_ModeB

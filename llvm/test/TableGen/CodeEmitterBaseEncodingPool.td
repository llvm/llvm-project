// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;

def Test : Target {
  let InstructionSet = TestInstrInfo;
}

class TestInst<bits<8> LowEncoding, bit HighEncoding> : Instruction {
  bits<128> Inst;
  let Inst{7...0} = LowEncoding;
  let Inst{64} = HighEncoding;
  let OutOperandList = (outs);
  let InOperandList = (ins);
}

def A0 : TestInst<1, 0>;
def A1 : TestInst<1, 0>;
def A2 : TestInst<1, 0>;
def A3 : TestInst<1, 0>;
def B0 : TestInst<1, 1>;
def B1 : TestInst<1, 1>;
def B2 : TestInst<1, 1>;
def B3 : TestInst<1, 1>;

// CHECK: static const uint64_t InstBits[][2] = {
// CHECK-NEXT: {UINT64_C(1), UINT64_C(0)},
// CHECK-NEXT: {UINT64_C(1), UINT64_C(1)},
// CHECK-NEXT: };
// CHECK: static const uint8_t InstBitsIndices[] = {
// CHECK-NEXT: 0,
// CHECK-NEXT: 0,
// CHECK-NEXT: 0,
// CHECK-NEXT: 0,
// CHECK-NEXT: 1,
// CHECK-NEXT: 1,
// CHECK-NEXT: 1,
// CHECK-NEXT: 1,
// CHECK-NEXT: };
// CHECK: static_assert(sizeof(InstBits) / sizeof(InstBits[0]) <=
// CHECK-NEXT: (UINT64_C(1) << (sizeof(InstBitsIndices[0]) * 8)));
// CHECK: unsigned InstBitsIndex = InstBitsIndices[TableIndex];
// CHECK: Inst = APInt(128, ArrayRef(InstBits[InstBitsIndex]));

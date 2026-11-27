// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o - 2>&1 >/dev/null | FileCheck %s

// A sub-register that is inherited from an explicit sub-register only keeps its
// index if the explicit sub-register is at offset 0 or at an unknown offset.
// Otherwise it is at a different offset in the super-register, so it needs a
// different index.

include "llvm/Target/Target.td"
def TestTarget : Target;

def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;
def lo32 : SubRegIndex<32, 0>;
def hi32 : SubRegIndex<32, 32>;
def any32 : SubRegIndex<32, -1>;
def hh16 : ComposedSubRegIndex<hi32, hi16>;

foreach N = ["A", "B", "C", "D"] in {
  def N#16L : Register<"">;
  def N#16H : Register<"">;
  def N#32 : Register<""> {
    let SubRegs = [!cast<Register>(N#"16L"), !cast<Register>(N#"16H")];
    let SubRegIndices = [lo16, hi16];
  }
}

// Sub-registers of the low half keep their indices, even though the high half
// is listed first. Sub-registers of the high half use a composed index if there
// is one, or a new index if not.
def AB64 : Register<""> {
  let SubRegs = [B32, A32];
  let SubRegIndices = [hi32, lo32];
}

// Sub-registers of a sub-register at an unknown offset keep their indices.
def C64 : Register<""> {
  let SubRegs = [C32];
  let SubRegIndices = [any32];
}

// The same applies if there is only a high half.
def D64 : Register<""> {
  let SubRegs = [D32];
  let SubRegIndices = [hi32];
}

def ABRC : RegisterClass<"Test", [i64], 0, (add AB64)>;
def CRC : RegisterClass<"Test", [i64], 0, (add C64)>;
def DRC : RegisterClass<"Test", [i64], 0, (add D64)>;

// CHECK-LABEL: SubRegIndex hh16:
// CHECK:         Offset: 48
// CHECK:         Size: 16

// CHECK-LABEL: SubRegIndex hi32_then_lo16:
// CHECK:         Offset: 32
// CHECK:         Size: 16

// CHECK-LABEL: Register AB64:
// CHECK:         SubReg hh16 = B16H
// CHECK-NEXT:    SubReg hi16 = A16H
// CHECK-NEXT:    SubReg hi32 = B32
// CHECK-NEXT:    SubReg lo16 = A16L
// CHECK-NEXT:    SubReg lo32 = A32
// CHECK-NEXT:    SubReg hi32_then_lo16 = B16L

// CHECK-LABEL: Register C64:
// CHECK:         SubReg any32 = C32
// CHECK-NEXT:    SubReg hi16 = C16H
// CHECK-NEXT:    SubReg lo16 = C16L

// CHECK-LABEL: Register D64:
// CHECK:         SubReg hh16 = D16H
// CHECK-NEXT:    SubReg hi32 = D32
// CHECK-NEXT:    SubReg hi32_then_lo16 = D16L

// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s | FileCheck %s

// Checks that identical super-register class masks are emitted once and
// shared, and that classes refer to them by offset.

include "llvm/Target/Target.td"

// Three levels of registers, one class per level:
//
//   Q0 -- D0 (sub0) -- S0 (ssub0)
//      \            \- S1 (ssub1)
//       \ D1 (sub1) -- S2 (ssub0)
//                   \- S3 (ssub1)

let Namespace = "MyTarget" in {
  def ssub0 : SubRegIndex<16>;
  def ssub1 : SubRegIndex<16, 16>;
  def sub0 : SubRegIndex<32>;
  def sub1 : SubRegIndex<32, 32>;

  def S0 : Register<"s0">;
  def S1 : Register<"s1">;
  def S2 : Register<"s2">;
  def S3 : Register<"s3">;

  let SubRegIndices = [ssub0, ssub1], CoveredBySubRegs = 1 in {
    def D0 : RegisterWithSubRegs<"d0", [S0, S1]>;
    def D1 : RegisterWithSubRegs<"d1", [S2, S3]>;
  }

  let SubRegIndices = [sub0, sub1], CoveredBySubRegs = 1 in
    def Q0 : RegisterWithSubRegs<"q0", [D0, D1]>;
}

def SRegs : RegisterClass<"MyTarget", [i16], 16, (sequence "S%u", 0, 3)>;
def DRegs : RegisterClass<"MyTarget", [i32], 32, (sequence "D%u", 0, 1)>;
def QRegs : RegisterClass<"MyTarget", [i64], 64, (add Q0)>;

def MyTarget : Target;

// The class IDs, and so the bit positions in the masks below, are
// SRegs = 0, DRegs = 1, QRegs = 2.

// SRegs has four super-register indices and DRegs has two, six masks in
// all, but only two of them are distinct. Both are emitted once, after
// the sub-class masks, and the offsets below point at those two.

// CHECK:      extern const MCRegisterClassStorage<3, {{[0-9]+}}, {{[0-9]+}}, 5, 6, {{[0-9]+}}, {{[0-9]+}}> MyTargetMCRegisterClassStorage = {

// CHECK:      // 0 SRegs
// CHECK-NEXT: 0x00000001,
// CHECK-NEXT: // 1 DRegs
// CHECK-NEXT: 0x00000002,
// CHECK-NEXT: // 2 QRegs
// CHECK-NEXT: 0x00000004,

// Shared by the two sub-register indices of SRegs that reach D registers,
// which are themselves covered by QRegs.
// CHECK-NEXT: /* 3 */ 0x00000006,

// Shared by the remaining two indices of SRegs and by both indices of
// DRegs, all of which reach Q registers only.
// CHECK-NEXT: /* 4 */ 0x00000004,
// CHECK-NEXT: },

// Each class names its masks by an offset relative to the class itself.
// CHECK:      /* 0 SRegs */
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[3]) - 0 * sizeof(MCRegisterClass),
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[3]) - 0 * sizeof(MCRegisterClass),
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[4]) - 0 * sizeof(MCRegisterClass),
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[4]) - 0 * sizeof(MCRegisterClass),
// CHECK-NEXT: /* 4 DRegs */
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[4]) - 1 * sizeof(MCRegisterClass),
// CHECK-SAME: offsetof(MyTargetRegisterClassStorage, SubClassMasks[4]) - 1 * sizeof(MCRegisterClass),

// QRegs has no super-registers and so no masks and no offsets.
// CHECK-NEXT: },

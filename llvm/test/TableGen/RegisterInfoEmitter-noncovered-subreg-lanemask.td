// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s | FileCheck %s

// Checks that computeRegUnitLaneMasks() handles noncovered subregs correctly.
include "llvm/Target/Target.td"

def TestTarget : Target;

def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;
def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;

// Fully-covered 32-bit register. This forces sub0/sub1 to be assigned
// lanes wide enough to cover both lo16 and hi16 of a 32-bit slot.
def R2_LO16 : Register<"r2l">;
def R2_HI16 : Register<"r2h">;
def R2 : RegisterWithSubRegs<"r2", [R2_LO16, R2_HI16]> {
  let SubRegIndices = [lo16, hi16];
  let CoveredBySubRegs = 1;
}

def R3_LO16 : Register<"r3l">;
def R3_HI16 : Register<"r3h">;
def R3 : RegisterWithSubRegs<"r3", [R3_LO16, R3_HI16]> {
  let SubRegIndices = [lo16, hi16];
  let CoveredBySubRegs = 1;
}

def R2R3 : RegisterWithSubRegs<"r2r3", [R2, R3]> {
  let SubRegIndices = [sub0, sub1];
  let CoveredBySubRegs = 1;
}

// Case with hi16 removed
def R0_LO16 : Register<"r0l">;
def R0 : RegisterWithSubRegs<"r0", [R0_LO16]> {
  let SubRegIndices = [lo16];
  let CoveredBySubRegs = 0;
}

def R1_LO16 : Register<"r1l">;
def R1 : RegisterWithSubRegs<"r1", [R1_LO16]> {
  let SubRegIndices = [lo16];
  let CoveredBySubRegs = 0;
}

def R0R1 : RegisterWithSubRegs<"r0r1", [R0, R1]> {
  let SubRegIndices = [sub0, sub1];
  let CoveredBySubRegs = 1;
}

def TestRC32 : RegisterClass<"Test", [i32], 32, (add R0, R1, R2, R3)>;
def TestRC64 : RegisterClass<"Test", [i64], 64, (add R0R1, R2R3)>;

// CHECK: extern const LaneBitmask TestTargetLaneMaskLists[] = {
// CHECK-NEXT: /* 0 */ LaneBitmask(0x0000000000000002), LaneBitmask(0x0000000000000001),
// CHECK-NEXT: /* 2 */ LaneBitmask(0x0000000000000002),
// CHECK-NEXT: /* 3 */ LaneBitmask(0x0000000000000002), LaneBitmask(0x0000000000000001), LaneBitmask(0x0000000000000004), LaneBitmask(0x0000000000000008),
// CHECK-NEXT: /* 7 */ LaneBitmask(0x0000000000000003), LaneBitmask(0x000000000000000C),
// CHECK-NEXT: /* 9 */ LaneBitmask(0xFFFFFFFFFFFFFFFF),
// CHECK-NEXT: };

// CHECK: extern const MCRegisterDesc TestTargetRegDesc[] = {
// CHECK: { 3, 26, 6, 3, 36864, 7, 0, 0 },

// CHECK: static const LaneBitmask TestTargetSubRegIndexLaneMaskTable[] = {
// CHECK-NEXT: LaneBitmask::getAll(),
// CHECK-NEXT: LaneBitmask(0x0000000000000001), // hi16
// CHECK-NEXT: LaneBitmask(0x0000000000000002), // lo16
// CHECK-NEXT: LaneBitmask(0x0000000000000003), // sub0
// CHECK-NEXT: LaneBitmask(0x000000000000000C), // sub1

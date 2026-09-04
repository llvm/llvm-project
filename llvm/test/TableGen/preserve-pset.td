// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s | FileCheck %s

// Verify that a marked singleton pressure set survives pruning while an
// overlapping larger set is retained too. Unmarked singleton pressure sets
// continue to use the normal pruning behavior.

include "llvm/Target/Target.td"

let Namespace = "TestTarget" in {
  def R0 : Register<"r0">;
  def R1 : Register<"r1">;
  def R2 : Register<"r2">;
}

def Singleton : RegisterClass<"TestTarget", [i32], 32, (add R0)> {
  let PreservePressureSet = true;
}

def PrunedSingleton : RegisterClass<"TestTarget", [i32], 32, (add R2)>;
def Overlapping : RegisterClass<"TestTarget", [i32], 32,
                                (add R0, R1, R2)>;

def TestTarget : Target;

// CHECK-LABEL: enum RegisterPressureSets {
// CHECK-NEXT: Singleton = 0,
// CHECK-NEXT: Overlapping = 1,
// CHECK-NEXT: };

// CHECK-LABEL: getRegPressureSetName(unsigned Idx) const {
// CHECK:         "Singleton",
// CHECK-NEXT:    "Overlapping",

// CHECK-LABEL: getRegPressureSetLimit(const MachineFunction &MF, unsigned Idx) const {
// CHECK:         1,  {{.*}}// 0: Singleton
// CHECK-NEXT:    3,  {{.*}}// 1: Overlapping

// Singleton maps to both its preserved dimension and the overlapping set.
// PrunedSingleton and Overlapping map only to the larger set.
// CHECK-LABEL: static const int RCSetsTable[] = {
// CHECK-NEXT:  /* 0 */ 0, 1, -1,
// CHECK-NEXT: };

// CHECK-LABEL: getRegClassPressureSets(const TargetRegisterClass *RC) const {
// CHECK: static const uint8_t RCSetStartTable[] = {
// Register class order is Overlapping, PrunedSingleton, Singleton. The first
// two use the suffix at offset 1, while Singleton uses both sets at offset 0.
// CHECK-NEXT: 1,1,0,};

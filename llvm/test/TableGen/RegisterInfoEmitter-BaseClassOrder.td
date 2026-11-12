// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

include "llvm/Target/Target.td"

let Namespace = "MyTarget" in {
  def R0 : Register<"r0">; // base class BaseA
  def R1 : Register<"r1">; // base class BaseA
  def R2 : Register<"r2">; // base class BaseC
  def R3 : Register<"r3">; // base class BaseC
  def R4 : Register<"r4">; // base class BaseB
  def R5 : Register<"r5">; // base class BaseB
  def R6 : Register<"r6">; // no base class
} // Namespace = "MyTarget"


// BaseA and BaseB are equal ordered so enumeration order determines base class for overlaps
def BaseA : RegisterClass<"MyTarget", [i32], 32, (sequence "R%u", 0, 3)> {
  let BaseClassOrder = 1;
}
def BaseB : RegisterClass<"MyTarget", [i32], 32, (sequence "R%u", 3, 5)> {
  let BaseClassOrder = 1;
}

// BaseC defined order overrides BaseA and BaseB
def BaseC : RegisterClass<"MyTarget", [i32], 32, (sequence "R%u", 2, 3)> {
  let BaseClassOrder = 0;
}

def MyTarget : Target;

// CHECK:      static const uint16_t Mapping[8] = {
// CHECK-NEXT:   InvalidRegClassID,  // NoRegister
// CHECK-NEXT:   MyTarget::BaseARegClassID,  // R0
// CHECK-NEXT:   MyTarget::BaseARegClassID,  // R1
// CHECK-NEXT:   MyTarget::BaseCRegClassID,  // R2
// CHECK-NEXT:   MyTarget::BaseCRegClassID,  // R3
// CHECK-NEXT:   MyTarget::BaseBRegClassID,  // R4
// CHECK-NEXT:   MyTarget::BaseBRegClassID,  // R5
// CHECK-NEXT:   InvalidRegClassID,  // R6
// CHECK-NEXT: };

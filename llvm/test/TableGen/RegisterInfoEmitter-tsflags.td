// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

// Configure and test TSFlags for a target.
include "llvm/Target/Target.td"

let Namespace = "MyTarget" in {
  def R : Register<"r">;
  def D : Register<"d">;
  def S : Register<"s">;
}

class MyClass <int size, list<ValueType> types, dag regList>
  : RegisterClass <"MyTarget", types, size, regList> {
  // Define the target bitfields.
  field bit A = 0;
  field bits<2> B = 0;

  // Associate the defined bitfields to unique bit positions in TSFlags.
  let TSFlags{0} = A;
  let TSFlags{2-1} = B;
}

// Default value for TSFlags.
def MyRegs : MyClass<32, [i32], (add R)>;

def SRegs : MyClass<32, [i32], (add S)> {
  let A = 1;
}

def DRegs : MyClass<32, [i32], (add D)>{
  let B = 3;
}

def SDRegs : MyClass<32, [i32], (add D, S)>{
  let A = 1;
  let B = 3;
}

def MyTarget : Target;

// CHECK:  extern const TargetRegisterClass SDRegsRegClass = {
// CHECK:  0x07, /* TSFlags */
// CHECK:  extern const TargetRegisterClass DRegsRegClass = {
// CHECK:  0x06, /* TSFlags */
// CHECK:  extern const TargetRegisterClass MyRegsRegClass = {
// CHECK:  0x00, /* TSFlags */
// CHECK:  extern const TargetRegisterClass SRegsRegClass = {
// CHECK:  0x01, /* TSFlags */

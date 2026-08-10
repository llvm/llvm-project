// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s 2>&1 | FileCheck %s

// Negative test to check empty Psets for a target.

include "llvm/Target/Target.td"

def R : Register<"r">;

def R_32 : RegisterClass<"MyTarget", [i32], 32, (add R)> {
  let GeneratePressureSet = 0;
}

def MyTarget : Target;

// CHECK: error: RegUnitSets cannot be empty!

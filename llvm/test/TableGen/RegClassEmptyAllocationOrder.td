// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %t \
// RUN:   %t/explicit.td -o %t/explicit.inc 2>&1 | FileCheck %t/explicit.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %t \
// RUN:   %t/inferred.td -o %t/inferred.inc 2>&1 | FileCheck %t/inferred.td --implicit-check-not="error:"
// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %t \
// RUN:   %t/no-members.td -o %t/no-members.inc 2>&1 | FileCheck %t/no-members.td --allow-empty --implicit-check-not="error:"

//--- Common.td
include "llvm/Target/Target.td"

class TestReg<string n> : Register<n> {
  let Namespace = "Test";
}

def R0 : TestReg<"r0">;
def R1 : TestReg<"r1">;
def R2 : TestReg<"r2">;

def TestTarget : Target;

//--- explicit.td
include "Common.td"

// CHECK: [[@LINE+1]]:5: error: allocatable register class 'RegsA' has an empty allocation order for AltOrder 0
def RegsA : RegisterClass<"Test", [i32], 32, (add R0, R1)> {
  let AltOrders = [(sub RegsA, R0, R1)];
  let AltOrderSelect = [{ return 1; }];
}

//--- inferred.td
include "Common.td"

// CHECK: error: allocatable register class 'Regs{{A_and_RegsB|B_and_RegsA}}' has an empty allocation order for AltOrder 0; define the inferred class explicitly
def RegsA : RegisterClass<"Test", [i32], 32, (add R0, R1)> {
  let AltOrders = [(add R0)];
  let AltOrderSelect = [{ return 1; }];
}

def RegsB : RegisterClass<"Test", [i32], 32, (add R1, R2)> {
  let AltOrders = [(add R2)];
  let AltOrderSelect = [{ return 1; }];
}

//--- no-members.td
include "Common.td"

def Regs : RegisterClass<"Test", [i32], 32, (add R0, R1)>;
def NoRegs : RegisterClass<"Test", [i32], 32, (add)>;

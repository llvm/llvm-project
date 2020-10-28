// RUN: not llvm-tblgen -gen-register-bank -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;
def R0 : Register<"r0">;
def ClassA : RegisterClass<"MyTarget", [], 32, (add R0)>; // CHECK: [[@LINE]]:1: error: RegTypes list must not be empty!

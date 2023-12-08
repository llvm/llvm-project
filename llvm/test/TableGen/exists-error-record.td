// RUN: not llvm-tblgen --no-warn-on-unused-template-args %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

//CHECK: expected string type argument in !exists operator, please use !isa instead

class A;
def a : A;
defvar value = !exists<A>(a);

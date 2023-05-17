// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A { int x; }

// CHECK: Undefined reference to record: ''
if !cast<A>("").x then
  def x;

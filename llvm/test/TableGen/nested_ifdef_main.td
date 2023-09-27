// RUN: llvm-tblgen -I %p %s | FileCheck %s

include "nested_ifdef.inc"
include "nested_ifdef2.inc"

// CHECK: def bar
// CHECK: def foo
// CHECK: def haha
// CHECK: def zoo

def zoo;

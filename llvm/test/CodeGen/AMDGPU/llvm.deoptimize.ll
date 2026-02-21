; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2>&1 | FileCheck %s

; CHECK: error: no deoptimize libcall available
declare i32 @llvm.experimental.deoptimize.i32(...)
declare i8  @llvm.experimental.deoptimize.i8(...)

define i32 @caller_0() {
entry:
  %v = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0) ]
  ret i32 %v
}


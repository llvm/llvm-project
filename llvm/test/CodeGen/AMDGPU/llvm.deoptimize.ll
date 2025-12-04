; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2> %t.err | FileCheck %s
; RUN: FileCheck -check-prefix=ERR %s < %t.err

; ERR: error: <unknown>:0:0: in function caller_0 i32 (): unsupported external symbol
; ERR: error: unhandled statepoint-like instruction

; CHECK: ;unsupported statepoint/stackmap/patchpoint
declare i32 @llvm.experimental.deoptimize.i32(...)
declare i8  @llvm.experimental.deoptimize.i8(...)

define i32 @caller_0() {
entry:
  %v = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0) ]
  ret i32 %v
}


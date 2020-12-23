; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s
; Test that statepoint intrinsic is marked with Throwable attribute and it is
; not optimized into call

declare i64 addrspace(1)* @gc_call()
declare token @llvm.experimental.gc.statepoint.p0f_p1i64f(i64, i32, i64 addrspace(1)* ()*, i32, i32, ...)
declare i32* @fake_personality_function()

define i32 @test() gc "statepoint-example" personality i32* ()* @fake_personality_function {
; CHECK-LABEL: test
entry:
  ; CHECK-LABEL: entry:
  ; CHECK-NEXT: %sp = invoke token (i64, i32, i64 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64f
  %sp = invoke token (i64, i32, i64 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64f(i64 0, i32 0, i64 addrspace(1)* ()* @gc_call, i32 0, i32 0, i32 0, i32 0)
                to label %normal unwind label %exception

exception:
  %lpad = landingpad { i8*, i32 }
          cleanup
  ret i32 0

normal:
  ret i32 1
}

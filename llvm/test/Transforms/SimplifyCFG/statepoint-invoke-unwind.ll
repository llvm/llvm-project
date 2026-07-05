; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s
; Test that statepoint intrinsic is marked with Throwable attribute and it is
; not optimized into call

declare ptr addrspace(1) @gc_call()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr addrspace(1) ()*, i32, i32, ...)
declare ptr @fake_personality_function()

define i32 @test() gc "statepoint-example" personality ptr @fake_personality_function {
; CHECK-LABEL: test
entry:
  ; CHECK-LABEL: entry:
  ; CHECK-NEXT: %sp = invoke token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0
  %sp = invoke token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(ptr addrspace(1) ()) @gc_call, i32 0, i32 0, i32 0, i32 0)
                to label %normal unwind label %exception

exception:
  %lpad = landingpad { ptr, i32 }
          cleanup
  ret i32 0

normal:
  ret i32 1
}

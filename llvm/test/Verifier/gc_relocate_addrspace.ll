; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; This is to verify that gc_relocate must return a pointer with the same
; address space with the relocated value.

; CHECK:       gc.relocate: relocating a pointer shouldn't change its address space
; CHECK-NEXT:  %obj.relocated = call coldcc i8* @llvm.experimental.gc.relocate.p0i8(token %safepoint_token, i32 7, i32 7) ;

declare void @foo()

; Function Attrs: nounwind
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...) #0

define void @test1(i64 addrspace(1)* %obj) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0, i64 addrspace(1)* %obj)
  %obj.relocated = call coldcc i8* @llvm.experimental.gc.relocate.p0i8(token %safepoint_token, i32 7, i32 7) ; (%obj, %obj)
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.experimental.gc.relocate.p0i8(token, i32, i32) #0

attributes #0 = { nounwind }

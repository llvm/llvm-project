; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; This is to verify that gc_relocate must return a pointer type, which is defined
; in intrinsics.td.

; CHECK: Intrinsic has incorrect return type!

declare void @foo()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

define void @test1(<2 x i32 addrspace(1)*> addrspace(1)* %obj) gc "statepoint-example" {
entry:
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0, <2 x i32 addrspace(1)*> addrspace(1)* %obj)
  %obj.relocated = call coldcc i8 @llvm.experimental.gc.relocate.i8(i32 %safepoint_token, i32 7, i32 7) ; (%obj, %obj)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.experimental.gc.relocate.i8(i32, i32, i32) #0

attributes #0 = { nounwind }


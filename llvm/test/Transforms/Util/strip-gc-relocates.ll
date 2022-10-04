; RUN: opt -S -strip-gc-relocates -instcombine < %s | FileCheck %s
; RUN: opt -S -passes=strip-gc-relocates,instcombine < %s | FileCheck %s
; test utility/debugging pass which removes gc.relocates, inserted by -rewrite-statepoints-for-gc
declare void @use_obj32(ptr addrspace(1)) "gc-leaf-function"

declare void @g()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32) #0
declare void @do_safepoint()

declare ptr addrspace(1) @new_instance() #1


; Simple case: remove gc.relocate
define ptr addrspace(1) @test1(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
; CHECK-LABEL: test1
; CHECK: gc.statepoint
; CHECK-NOT: gc.relocate
; CHECK: ret ptr addrspace(1) %arg
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @g, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg), "deopt" (i32 100)]
  %arg.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 0, i32 0) ; (%arg, %arg)
  ret ptr addrspace(1) %arg.relocated
}

; Remove gc.relocates in presence of nested relocates.
define void @test2(ptr addrspace(1) %base) gc "statepoint-example" {
entry:
; CHECK-LABEL: test2
; CHECK: statepoint
; CHECK-NOT: gc.relocate
; CHECK: call void @use_obj32(ptr addrspace(1) %ptr.gep1)
; CHECK: call void @use_obj32(ptr addrspace(1) %ptr.gep1)
  %ptr.gep = getelementptr i32, ptr addrspace(1) %base, i32 15
  %ptr.gep1 = getelementptr i32, ptr addrspace(1) %ptr.gep, i32 15
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %ptr.gep1, ptr addrspace(1) %base)]
  %ptr.gep1.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 1, i32 0) ; (%base, %ptr.gep1)
  %base.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 1, i32 1) ; (%base, %base)
  call void @use_obj32(ptr addrspace(1) %ptr.gep1.relocated)
  %statepoint_token1 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %ptr.gep1.relocated, ptr addrspace(1) %base.relocated)]
  %ptr.gep1.relocated2 = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token1, i32 1, i32 0) ; (%base.relocated, %ptr.gep1.relocated)
  %base.relocated3 = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token1, i32 1, i32 1) ; (%base.relocated, %base.relocated)
  call void @use_obj32(ptr addrspace(1) %ptr.gep1.relocated2)
  ret void
}

; landing pad gc.relocates removed by instcombine since it has no uses.
define ptr addrspace(1) @test3(ptr addrspace(1) %arg) gc "statepoint-example" personality i32 8 {
; CHECK-LABEL: test3(
; CHECK: gc.statepoint
; CHECK-LABEL: normal_dest:
; CHECK-NOT: gc.relocate
; CHECK: ret ptr addrspace(1) %arg
; CHECK-LABEL: unwind_dest:
; CHECK-NOT: gc.relocate
entry:
  %statepoint_token = invoke token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @g, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg), "deopt" (i32 100)]
          to label %normal_dest unwind label %unwind_dest

normal_dest:                                      ; preds = %entry
  %arg.relocated1 = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 0, i32 0) ; (%arg, %arg)
  ret ptr addrspace(1) %arg.relocated1

unwind_dest:                                      ; preds = %entry
  %lpad = landingpad token
          cleanup
  %arg.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %lpad, i32 0, i32 0) ; (%arg, %arg)
  resume token undef
}

; in presence of phi
define void @test4(i1 %cond) gc "statepoint-example" {
; CHECK-LABEL: test4
entry:
  %base1 = call ptr addrspace(1) @new_instance()
  %base2 = call ptr addrspace(1) @new_instance()
  br i1 %cond, label %here, label %there

here:                                             ; preds = %entry
  br label %merge

there:                                            ; preds = %entry
  br label %merge

merge:                                            ; preds = %there, %here
; CHECK-LABEL: merge:
; CHECK-NOT: gc.relocate
; CHECK: %ptr.gep.remat = getelementptr i32, ptr addrspace(1) %basephi.base
  %basephi.base = phi ptr addrspace(1) [ %base1, %here ], [ %base2, %there ], !is_base_value !0
  %basephi = phi ptr addrspace(1) [ %base1, %here ], [ %base2, %there ]
  %ptr.gep = getelementptr i32, ptr addrspace(1) %basephi, i32 15
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %basephi.base)]
  %basephi.base.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 0, i32 0) ; (%basephi.base, %basephi.base)
  %ptr.gep.remat = getelementptr i32, ptr addrspace(1) %basephi.base.relocated, i32 15
  call void @use_obj32(ptr addrspace(1) %ptr.gep.remat)
  ret void
}

; The gc.relocate type is different from %arg, but removing the gc.relocate,
; needs a bitcast to be added from ptr addrspace(1) to ptr addrspace(1)
define ptr addrspace(1) @test5(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
; CHECK-LABEL: test5
; CHECK: gc.statepoint
; CHECK-NOT: gc.relocate
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @g, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg), "deopt" (i32 100)]
  %arg.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %statepoint_token, i32 0, i32 0) ; (%arg, %arg)
  ret ptr addrspace(1) %arg.relocated
}

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind "gc-leaf-function" }
!0 = !{}

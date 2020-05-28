; RUN: opt -S -strip-gc-relocates -instcombine < %s | FileCheck %s
; test utility/debugging pass which removes gc.relocates, inserted by -rewrite-statepoints-for-gc
declare void @use_obj32(i32 addrspace(1)*) "gc-leaf-function"

declare void @g()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32) #0
declare void @do_safepoint()

declare i32 addrspace(1)* @new_instance() #1


; Simple case: remove gc.relocate
define i32 addrspace(1)* @test1(i32 addrspace(1)* %arg) gc "statepoint-example" {
entry:
; CHECK-LABEL: test1
; CHECK: gc.statepoint
; CHECK-NOT: gc.relocate
; CHECK: ret i32 addrspace(1)* %arg
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @g, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %arg) ["deopt" (i32 100)]
  %arg.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7) ; (%arg, %arg)
  %arg.relocated.casted = bitcast i8 addrspace(1)* %arg.relocated to i32 addrspace(1)*
  ret i32 addrspace(1)* %arg.relocated.casted
}

; Remove gc.relocates in presence of nested relocates.
define void @test2(i32 addrspace(1)* %base) gc "statepoint-example" {
entry:
; CHECK-LABEL: test2
; CHECK: statepoint
; CHECK-NOT: gc.relocate
; CHECK: call void @use_obj32(i32 addrspace(1)* %ptr.gep1)
; CHECK: call void @use_obj32(i32 addrspace(1)* %ptr.gep1)
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.gep1 = getelementptr i32, i32 addrspace(1)* %ptr.gep, i32 15
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %ptr.gep1, i32 addrspace(1)* %base)
  %ptr.gep1.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 8, i32 7) ; (%base, %ptr.gep1)
  %ptr.gep1.relocated.casted = bitcast i8 addrspace(1)* %ptr.gep1.relocated to i32 addrspace(1)*
  %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 8, i32 8) ; (%base, %base)
  %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %ptr.gep1.relocated.casted)
  %statepoint_token1 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %ptr.gep1.relocated.casted, i32 addrspace(1)* %base.relocated.casted)
  %ptr.gep1.relocated2 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1, i32 8, i32 7) ; (%base.relocated.casted, %ptr.gep1.relocated.casted)
  %ptr.gep1.relocated2.casted = bitcast i8 addrspace(1)* %ptr.gep1.relocated2 to i32 addrspace(1)*
  %base.relocated3 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1, i32 8, i32 8) ; (%base.relocated.casted, %base.relocated.casted)
  %base.relocated3.casted = bitcast i8 addrspace(1)* %base.relocated3 to i32 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %ptr.gep1.relocated2.casted)
  ret void
}

; landing pad gc.relocates removed by instcombine since it has no uses.
define i32 addrspace(1)* @test3(i32 addrspace(1)* %arg) gc "statepoint-example" personality i32 8 {
; CHECK-LABEL: test3(
; CHECK: gc.statepoint
; CHECK-LABEL: normal_dest:
; CHECK-NOT: gc.relocate
; CHECK: ret i32 addrspace(1)* %arg
; CHECK-LABEL: unwind_dest:
; CHECK-NOT: gc.relocate
entry:
  %statepoint_token = invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @g, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %arg) ["deopt" (i32 100)]
          to label %normal_dest unwind label %unwind_dest

normal_dest:                                      ; preds = %entry
  %arg.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7) ; (%arg, %arg)
  %arg.relocated1.casted = bitcast i8 addrspace(1)* %arg.relocated1 to i32 addrspace(1)*
  ret i32 addrspace(1)* %arg.relocated1.casted

unwind_dest:                                      ; preds = %entry
  %lpad = landingpad token
          cleanup
  %arg.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %lpad, i32 7, i32 7) ; (%arg, %arg)
  %arg.relocated.casted = bitcast i8 addrspace(1)* %arg.relocated to i32 addrspace(1)*
  resume token undef
}

; in presence of phi
define void @test4(i1 %cond) gc "statepoint-example" {
; CHECK-LABEL: test4
entry:
  %base1 = call i32 addrspace(1)* @new_instance()
  %base2 = call i32 addrspace(1)* @new_instance()
  br i1 %cond, label %here, label %there

here:                                             ; preds = %entry
  br label %merge

there:                                            ; preds = %entry
  br label %merge

merge:                                            ; preds = %there, %here
; CHECK-LABEL: merge:
; CHECK-NOT: gc.relocate
; CHECK: %ptr.gep.remat = getelementptr i32, i32 addrspace(1)* %basephi.base
  %basephi.base = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ], !is_base_value !0
  %basephi = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ]
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %basephi, i32 15
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %basephi.base)
  %basephi.base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7) ; (%basephi.base, %basephi.base)
  %basephi.base.relocated.casted = bitcast i8 addrspace(1)* %basephi.base.relocated to i32 addrspace(1)*
  %ptr.gep.remat = getelementptr i32, i32 addrspace(1)* %basephi.base.relocated.casted, i32 15
  call void @use_obj32(i32 addrspace(1)* %ptr.gep.remat)
  ret void
}

; The gc.relocate type is different from %arg, but removing the gc.relocate,
; needs a bitcast to be added from i32 addrspace(1)* to i8 addrspace(1)*
define i8 addrspace(1)* @test5(i32 addrspace(1)* %arg) gc "statepoint-example" {
entry:
; CHECK-LABEL: test5
; CHECK: gc.statepoint
; CHECK-NOT: gc.relocate
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @g, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %arg) ["deopt" (i32 100)]
  %arg.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7) ; (%arg, %arg)
  ret i8 addrspace(1)* %arg.relocated
}

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind "gc-leaf-function" }
!0 = !{}

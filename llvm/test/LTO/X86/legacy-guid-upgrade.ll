;; Test that regular LTO handles bitcode produced before GUID assignment
;; existed (i.e. without a MODULE_CODE_GUIDLIST record and !guid metadata).
;; GUIDs are assigned on load from the original names and linkages, so whole
;; program devirtualization can look up the (promoted and renamed) internal
;; virtual function in the summary when merged with a split ThinLTO module.
;;
;; Inputs/legacy-guid-upgrade-wpd.bc was produced from the regular LTO module
;; below with an LLVM 21 opt -module-summary.

; RUN: split-file %s %t
; RUN: opt --thinlto-bc --thinlto-split-lto-unit %t/thin.ll -o %t/thin.bc
; RUN: llvm-lto2 run %t/thin.bc %p/Inputs/legacy-guid-upgrade-wpd.bc -o %t/out \
; RUN:   -save-temps -pass-remarks=wholeprogramdevirt \
; RUN:   -r=%t/thin.bc,_start,plx \
; RUN:   -r=%p/Inputs/legacy-guid-upgrade-wpd.bc,vt,pl \
; RUN:   -r=%p/Inputs/legacy-guid-upgrade-wpd.bc,call,plx 2>&1 | FileCheck %s
; RUN: llvm-dis %t/out.0.4.opt.bc -o - | FileCheck %s --check-prefix=IR

; CHECK-DAG: single-impl: devirtualized a call to vf.llvm.merged
; CHECK-DAG: devirtualized vf

; IR: define hidden void @vf.llvm.merged(ptr {{.*}}%obj) {{.*}} !guid !{{[0-9]+}} {

;--- thin.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"T")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

;--- legacy-guid-upgrade-wpd.ll
source_filename = "legacy-wpd.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vt = constant ptr @vf, !type !0, !vcall_visibility !1
@llvm.used = appending global [1 x ptr] [ptr @vt], section "llvm.metadata"

define internal void @vf(ptr %obj) {
  ret void
}

define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"T")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!llvm.module.flags = !{!2, !3}

!0 = !{i64 0, !"T"}
!1 = !{i64 1}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

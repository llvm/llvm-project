; RUN: opt -passes=lower-comment-string -S %s -o - | FileCheck %s

; Verify that lower-comment-string is enabled by default on all opt pipelines.
; RUN: opt --O0 -S %s -o - | FileCheck %s
; RUN: opt --O1 -S %s -o - | FileCheck %s
; RUN: opt --O2 -S %s -o - | FileCheck %s
; RUN: opt --O3 -S %s -o - | FileCheck %s

target triple = "powerpc-ibm-aix"

@__loadtime_comment_str_f20696a95b638f0b = weak_odr hidden unnamed_addr constant [24 x i8] c"@(#) Copyright TU1 v1.0\00", section "__loadtime_comment", align 1, !loadtime_comment !0
@.loadtime_comment_vars.str = private unnamed_addr constant [22 x i8] c"loadtime_comment vars\00", align 1
@loadtime_comment_vars_gv = internal global ptr @.loadtime_comment_vars.str, align 4, !loadtime_comment !0
@llvm.compiler.used = appending global [2 x ptr] [ptr @__loadtime_comment_str_f20696a95b638f0b, ptr @loadtime_comment_vars_gv], section "llvm.metadata"

define void @f0() {
entry:
  ret void
}
define i32 @main() {
entry:
  ret i32 0
}

!0 = !{}
; ---- Globals --------------------------------------------
; CHECK: @[[LOADTIME_COMMENT_STR:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [24 x i8] c"@(#) Copyright TU1 v1.0\00", section "__loadtime_comment", align 1, !loadtime_comment !0
; CHECK: @.loadtime_comment_vars.str = private unnamed_addr constant [22 x i8] c"loadtime_comment vars\00", align 1
; CHECK: @loadtime_comment_vars_gv = internal global ptr @.loadtime_comment_vars.str, align {{[0-9]+}}, !loadtime_comment !0
; CHECK-NEXT: @llvm.compiler.used = appending global [2 x ptr] [ptr @[[LOADTIME_COMMENT_STR]], ptr @loadtime_comment_vars_gv], section "llvm.metadata"


; Function has implicit refs to both loadtime comment globals.
; CHECK: define void @f0()
; CHECK-SAME: !implicit.ref ![[MD:[0-9]+]]
; CHECK-SAME: !implicit.ref ![[MD2:[0-9]+]]
; CHECK: define {{.*}}i32 @main()
; CHECK-SAME: !implicit.ref ![[MD]]
; CHECK-SAME: !implicit.ref ![[MD2]]

; Verify metadata content
; CHECK: ![[MD]] = !{ptr @[[LOADTIME_COMMENT_STR]]}
; CHECK: ![[MD2]] = !{ptr @loadtime_comment_vars_gv}

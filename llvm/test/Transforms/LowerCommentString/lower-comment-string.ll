; RUN: opt -passes=lower-comment-string -S %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O0

; Verify that lower-comment-string is enabled by default on all opt pipelines.
; RUN: opt --O0 -S %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O0
; RUN: opt --O1 -S %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON
; RUN: opt --O2 -S %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON
; RUN: opt --O3 -S %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ON

target triple = "powerpc-ibm-aix"

@__loadtime_comment_str_f20696a95b638f0b = weak_odr hidden unnamed_addr constant [24 x i8] c"@(#) Copyright TU1 v1.0\00", section "__loadtime_comment", align 1, !loadtime_comment !0
@.loadtime_comment_vars.str = private unnamed_addr constant [22 x i8] c"loadtime_comment vars\00", align 1
@loadtime_comment_vars_gv = internal global ptr @.loadtime_comment_vars.str, align 8, !loadtime_comment !0
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
; CHECK-O0: define void @f0() !implicit.ref ![[MD:[0-9]+]] !implicit.ref ![[MD2:[0-9]+]]
; CHECK-ON: define void @f0() local_unnamed_addr #0 !implicit.ref ![[MD:[0-9]+]] !implicit.ref ![[MD2:[0-9]+]]
; CHECK-O0: define i32 @main() !implicit.ref ![[MD]] !implicit.ref ![[MD2]]
; CHECK-ON: define noundef i32 @main() local_unnamed_addr #0 !implicit.ref ![[MD]] !implicit.ref ![[MD2]]

; Verify metadata content
; CHECK-O0: ![[MD]] = !{ptr @[[LOADTIME_COMMENT_STR]]}
; CHECK-ON: ![[MD]] = !{ptr @[[LOADTIME_COMMENT_STR]]}
; CHECK-O0: ![[MD2]] = !{ptr @loadtime_comment_vars_gv}
; CHECK-ON: ![[MD2]] = !{ptr @loadtime_comment_vars_gv}

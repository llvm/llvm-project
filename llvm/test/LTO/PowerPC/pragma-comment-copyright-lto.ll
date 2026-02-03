; RUN: rm -rf %t && mkdir %t
; RUN: split-file %s %t
; RUN: llvm-as %t/tu1.ll -o %t/tu1.bc
; RUN: llvm-as %t/tu2.ll -o %t/tu2.bc
; RUN: llvm-lto -filetype=asm \
; RUN:   -exported-symbol=main \
; RUN:   -exported-symbol=f_tu1 \
; RUN:   %t/tu1.bc %t/tu2.bc \
; RUN:   -o %t/out.s 
; RUN: FileCheck %s < %t/out.s


;--- tu1.ll
;; TU1: already lowered by LowerCommentStringPass at prelink
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

@__loadtime_comment_str = internal unnamed_addr constant
    [14 x i8] c"Copyright TU1\00",
    section "__loadtime_comment", align 1
@llvm.compiler.used = appending global [1 x ptr]
    [ptr @__loadtime_comment_str], section "llvm.metadata"

define void @f_tu1() !implicit.ref !0 {
entry:
  ret void
}

!0 = !{ptr @__loadtime_comment_str}

;--- tu2.ll
;; TU2: already lowered by LowerCommentStringPass at prelink
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

@__loadtime_comment_str = internal unnamed_addr constant
    [14 x i8] c"Copyright TU2\00",
    section "__loadtime_comment", align 1
@llvm.compiler.used = appending global [1 x ptr]
    [ptr @__loadtime_comment_str], section "llvm.metadata"

declare void @f_tu1()

define i32 @main() !implicit.ref !0 {
entry:
  call void @f_tu1()
  ret i32 0
}

!0 = !{ptr @__loadtime_comment_str}

;; .ref directive emitted from f_tu1's csect anchoring TU1's string
; CHECK-LABEL: .f_tu1:
; CHECK-NEXT:  .ref __loadtime_comment_str

;; .ref directive emitted from main's csect anchoring TU2's string
; CHECK-LABEL: .main:
; CHECK-NEXT:  .ref __loadtime_comment_str.2

;; Both copyright strings in the read-only __loadtime_comment csect
; CHECK:      .csect __loadtime_comment[RO],2
; CHECK-DAG:   .lglobl __loadtime_comment_str
; CHECK-LABEL: __loadtime_comment_str:
; CHECK-DAG:  .string "Copyright TU1"
; CHECK-DAG:   .lglobl __loadtime_comment_str.2
; CHECK-LABEL: __loadtime_comment_str.2:
; CHECK-DAG:  .string "Copyright TU2"

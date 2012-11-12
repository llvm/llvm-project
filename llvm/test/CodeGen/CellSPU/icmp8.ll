; RUN: llc < %s -march=cellspu | FileCheck %s

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; $3 = %arg1, $4 = %arg2, $5 = %val1, $6 = %val2
; $3 = %arg1, $4 = %val1, $5 = %val2
;
; For "positive" comparisons:
; selb $3, $6, $5, <i1>
; selb $3, $5, $4, <i1>
;
; For "negative" comparisons, i.e., those where the result of the comparison
; must be inverted (setne, for example):
; selb $3, $5, $6, <i1>
; selb $3, $4, $5, <i1>

; i8 integer comparisons:
define i8 @icmp_eq_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_eq_select_i8:
; CHECK:        ceqb
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp eq i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_eq_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_eq_setcc_i8:
; CHECK:        ceqb
; CHECK-NEXT:   bi

entry:
       %A = icmp eq i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_eq_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_eq_immed01_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i8 %arg1, 127
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_eq_immed02_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_eq_immed02_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i8 %arg1, -128
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_eq_immed03_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_eq_immed03_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i8 %arg1, -1
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ne_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ne_select_i8:
; CHECK:        ceqb
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp ne i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_ne_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ne_setcc_i8:
; CHECK:        ceqb
; CHECK:        xorbi
; CHECK-NEXT:   bi

entry:
       %A = icmp ne i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_ne_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ne_immed01_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i8 %arg1, 127
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ne_immed02_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ne_immed02_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i8 %arg1, -128
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ne_immed03_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ne_immed03_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i8 %arg1, -1
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ugt_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ugt_select_i8:
; CHECK:        clgtb
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp ugt i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_ugt_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ugt_setcc_i8:
; CHECK:        clgtb
; CHECK-NEXT:   bi

entry:
       %A = icmp ugt i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_ugt_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ugt_immed01_i8:
; CHECK:        clgtbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ugt i8 %arg1, 126
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_uge_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_uge_select_i8:
; CHECK:        ceqb
; CHECK:        clgtb
; CHECK:        or
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp uge i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_uge_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_uge_setcc_i8:
; CHECK:        ceqb
; CHECK:        clgtb
; CHECK:        or
; CHECK-NEXT:   bi

entry:
       %A = icmp uge i8 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp uge i8 %arg1, <immed> can always be transformed into
;;       icmp ugt i8 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i8 @icmp_ult_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ult_select_i8:
; CHECK:        ceqb
; CHECK:        clgtb
; CHECK:        nor
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp ult i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_ult_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ult_setcc_i8:
; CHECK:        ceqb
; CHECK:        clgtb
; CHECK:        nor
; CHECK-NEXT:   bi

entry:
       %A = icmp ult i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_ult_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ult_immed01_i8:
; CHECK:        ceqbi
; CHECK:        clgtbi
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i8 %arg1, 253
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ult_immed02_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ult_immed02_i8:
; CHECK:        ceqbi
; CHECK:        clgtbi
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i8 %arg1, 129
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_ule_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ule_select_i8:
; CHECK:        clgtb
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp ule i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_ule_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_ule_setcc_i8:
; CHECK:        clgtb
; CHECK:        xorbi
; CHECK-NEXT:   bi

entry:
       %A = icmp ule i8 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp ule i8 %arg1, <immed> can always be transformed into
;;       icmp ult i8 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i8 @icmp_sgt_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sgt_select_i8:
; CHECK:        cgtb
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp sgt i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_sgt_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sgt_setcc_i8:
; CHECK:        cgtb
; CHECK-NEXT:   bi

entry:
       %A = icmp sgt i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_sgt_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sgt_immed01_i8:
; CHECK:        cgtbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i8 %arg1, 96
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_sgt_immed02_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sgt_immed02_i8:
; CHECK:        cgtbi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i8 %arg1, -1
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_sgt_immed03_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sgt_immed03_i8:
; CHECK:        ceqbi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp sgt i8 %arg1, -128
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_sge_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sge_select_i8:
; CHECK:        ceqb
; CHECK:        cgtb
; CHECK:        or
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp sge i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_sge_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sge_setcc_i8:
; CHECK:        ceqb
; CHECK:        cgtb
; CHECK:        or
; CHECK-NEXT:   bi

entry:
       %A = icmp sge i8 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sge i8 %arg1, <immed> can always be transformed into
;;       icmp sgt i8 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i8 @icmp_slt_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_slt_select_i8:
; CHECK:        ceqb
; CHECK:        cgtb
; CHECK:        nor
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp slt i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_slt_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_slt_setcc_i8:
; CHECK:        ceqb
; CHECK:        cgtb
; CHECK:        nor
; CHECK-NEXT:   bi

entry:
       %A = icmp slt i8 %arg1, %arg2
       ret i1 %A
}

define i8 @icmp_slt_immed01_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_slt_immed01_i8:
; CHECK:        ceqbi
; CHECK:        cgtbi
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i8 %arg1, 96
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_slt_immed02_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_slt_immed02_i8:
; CHECK:        ceqbi
; CHECK:        cgtbi
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i8 %arg1, -120
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_slt_immed03_i8(i8 %arg1, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_slt_immed03_i8:
; CHECK:        ceqbi
; CHECK:        cgtbi
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i8 %arg1, -1
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i8 @icmp_sle_select_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sle_select_i8:
; CHECK:        cgtb
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp sle i8 %arg1, %arg2
       %B = select i1 %A, i8 %val1, i8 %val2
       ret i8 %B
}

define i1 @icmp_sle_setcc_i8(i8 %arg1, i8 %arg2, i8 %val1, i8 %val2) nounwind {
; CHECK:      icmp_sle_setcc_i8:
; CHECK:        cgtb
; CHECK:        xorbi
; CHECK-NEXT:   bi

entry:
       %A = icmp sle i8 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sle i8 %arg1, <immed> can always be transformed into
;;       icmp slt i8 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.


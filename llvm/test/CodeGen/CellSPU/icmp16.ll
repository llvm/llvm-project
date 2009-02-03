; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep ilh                                %t1.s | count 5
; RUN: grep ceqh                               %t1.s | count 29
; RUN: grep ceqhi                              %t1.s | count 13
; RUN: grep clgth                              %t1.s | count 15
; RUN: grep cgth                               %t1.s | count 14
; RUN: grep cgthi                              %t1.s | count 6
; RUN: grep {selb\t\\\$3, \\\$6, \\\$5, \\\$3} %t1.s | count 7
; RUN: grep {selb\t\\\$3, \\\$5, \\\$6, \\\$3} %t1.s | count 3
; RUN: grep {selb\t\\\$3, \\\$5, \\\$4, \\\$3} %t1.s | count 17
; RUN: grep {selb\t\\\$3, \\\$4, \\\$5, \\\$3} %t1.s | count 6

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

; i16 integer comparisons:
define i16 @icmp_eq_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_eq_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_eq_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, 511
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_eq_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, -512
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_eq_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, -1
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_eq_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp eq i16 %arg1, 32768
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ne_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_ne_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_ne_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, 511
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ne_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, -512
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ne_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, -1
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ne_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ne i16 %arg1, 32768
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ugt_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_ugt_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_ugt_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, 500
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ugt_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, 0
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ugt_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, 65024
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ugt_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ugt i16 %arg1, 32768
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_uge_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp uge i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_uge_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp uge i16 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp uge i16 %arg1, <immed> can always be transformed into
;;       icmp ugt i16 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i16 @icmp_ult_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_ult_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_ult_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, 511
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ult_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, 65534
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ult_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, 65024
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ult_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ult i16 %arg1, 32769
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_ule_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ule i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_ule_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp ule i16 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp ule i16 %arg1, <immed> can always be transformed into
;;       icmp ult i16 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i16 @icmp_sgt_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_sgt_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_sgt_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, 511
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_sgt_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, -1
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_sgt_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, -512
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_sgt_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sgt i16 %arg1, 32768
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_sge_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sge i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_sge_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sge i16 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sge i16 %arg1, <immed> can always be transformed into
;;       icmp sgt i16 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i16 @icmp_slt_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_slt_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, %arg2
       ret i1 %A
}

define i16 @icmp_slt_immed01_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, 511
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_slt_immed02_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, -512
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_slt_immed03_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, -1
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_slt_immed04_i16(i16 %arg1, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp slt i16 %arg1, 32768
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i16 @icmp_sle_select_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sle i16 %arg1, %arg2
       %B = select i1 %A, i16 %val1, i16 %val2
       ret i16 %B
}

define i1 @icmp_sle_setcc_i16(i16 %arg1, i16 %arg2, i16 %val1, i16 %val2) nounwind {
entry:
       %A = icmp sle i16 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sle i16 %arg1, <immed> can always be transformed into
;;       icmp slt i16 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.


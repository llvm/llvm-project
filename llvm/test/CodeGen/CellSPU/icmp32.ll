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

; i32 integer comparisons:
define i32 @icmp_eq_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_select_i32:
; CHECK:        ceq
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp eq i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_eq_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_setcc_i32:
; CHECK:        ilhu
; CHECK:        ceq
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp eq i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_eq_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_immed01_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_eq_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_immed02_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i32 %arg1, -512
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_eq_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_immed03_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i32 %arg1, -1
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_eq_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_eq_immed04_i32:
; CHECK:        ila
; CHECK:        ceq
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp eq i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ne_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_select_i32:
; CHECK:        ceq
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp ne i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_ne_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_setcc_i32:
; CHECK:        ceq
; CHECK:        ilhu
; CHECK:        xori
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp ne i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_ne_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_immed01_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ne_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_immed02_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i32 %arg1, -512
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ne_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_immed03_i32:
; CHECK:        ceqi
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i32 %arg1, -1
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ne_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ne_immed04_i32:
; CHECK:        ila
; CHECK:        ceq
; CHECK:        selb $3, $4, $5, $3

entry:
       %A = icmp ne i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ugt_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_select_i32:
; CHECK:        clgt
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp ugt i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_ugt_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_setcc_i32:
; CHECK:        ilhu
; CHECK:        clgt
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp ugt i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_ugt_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_immed01_i32:
; CHECK:        clgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ugt i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ugt_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_immed02_i32:
; CHECK:        clgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ugt i32 %arg1, 4294966784
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ugt_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_immed03_i32:
; CHECK:        clgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ugt i32 %arg1, 4294967293
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ugt_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ugt_immed04_i32:
; CHECK:        ila
; CHECK:        clgt
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ugt i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_uge_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_uge_select_i32:
; CHECK:        ceq
; CHECK:        clgt
; CHECK:        or
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp uge i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_uge_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_uge_setcc_i32:
; CHECK:        ceq
; CHECK:        clgt
; CHECK:        ilhu
; CHECK:        or
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp uge i32 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp uge i32 %arg1, <immed> can always be transformed into
;;       icmp ugt i32 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i32 @icmp_ult_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_select_i32:
; CHECK:        ceq
; CHECK:        clgt
; CHECK:        nor
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp ult i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_ult_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_setcc_i32:
; CHECK:        ceq
; CHECK:        clgt
; CHECK:        ilhu
; CHECK:        nor
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp ult i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_ult_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_immed01_i32:
; CHECK:        ceqi
; CHECK:        clgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ult_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_immed02_i32:
; CHECK:        ceqi
; CHECK:        clgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i32 %arg1, 4294966784
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ult_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_immed03_i32:
; CHECK:        ceqi
; CHECK:        clgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i32 %arg1, 4294967293
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ult_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ult_immed04_i32:
; CHECK:        rotmi
; CHECK:        ceqi
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp ult i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_ule_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ule_select_i32:
; CHECK:        clgt
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp ule i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_ule_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_ule_setcc_i32:
; CHECK:        clgt
; CHECK:        ilhu
; CHECK:        xori
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp ule i32 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp ule i32 %arg1, <immed> can always be transformed into
;;       icmp ult i32 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i32 @icmp_sgt_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_select_i32:
; CHECK:        cgt
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp sgt i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_sgt_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_setcc_i32:
; CHECK:        ilhu
; CHECK:        cgt
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp sgt i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_sgt_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_immed01_i32:
; CHECK:        cgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_sgt_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_immed02_i32:
; CHECK:        cgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i32 %arg1, 4294966784
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_sgt_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_immed03_i32:
; CHECK:        cgti
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i32 %arg1, 4294967293
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_sgt_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sgt_immed04_i32:
; CHECK:        ila
; CHECK:        cgt
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp sgt i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_sge_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sge_select_i32:
; CHECK:        ceq
; CHECK:        cgt
; CHECK:        or
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp sge i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_sge_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sge_setcc_i32:
; CHECK:        ceq
; CHECK:        cgt
; CHECK:        ilhu
; CHECK:        or
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp sge i32 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sge i32 %arg1, <immed> can always be transformed into
;;       icmp sgt i32 %arg1, <immed>-1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.

define i32 @icmp_slt_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_select_i32:
; CHECK:        ceq
; CHECK:        cgt
; CHECK:        nor
; CHECK:        selb $3, $6, $5, $3

entry:
       %A = icmp slt i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_slt_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_setcc_i32:
; CHECK:        ceq
; CHECK:        cgt
; CHECK:        ilhu
; CHECK:        nor
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp slt i32 %arg1, %arg2
       ret i1 %A
}

define i32 @icmp_slt_immed01_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_immed01_i32:
; CHECK:        ceqi
; CHECK:        cgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i32 %arg1, 511
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_slt_immed02_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_immed02_i32:
; CHECK:        ceqi
; CHECK:        cgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i32 %arg1, -512
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_slt_immed03_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_immed03_i32:
; CHECK:        ceqi
; CHECK:        cgti
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i32 %arg1, -1
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_slt_immed04_i32(i32 %arg1, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_slt_immed04_i32:
; CHECK:        ila
; CHECK:        ceq
; CHECK:        cgt
; CHECK:        nor
; CHECK:        selb $3, $5, $4, $3

entry:
       %A = icmp slt i32 %arg1, 32768
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i32 @icmp_sle_select_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sle_select_i32:
; CHECK:        cgt
; CHECK:        selb $3, $5, $6, $3

entry:
       %A = icmp sle i32 %arg1, %arg2
       %B = select i1 %A, i32 %val1, i32 %val2
       ret i32 %B
}

define i1 @icmp_sle_setcc_i32(i32 %arg1, i32 %arg2, i32 %val1, i32 %val2) nounwind {
; CHECK:      icmp_sle_setcc_i32:
; CHECK:        cgt
; CHECK:        ilhu
; CHECK:        xori
; CHECK:        iohl
; CHECK:        shufb

entry:
       %A = icmp sle i32 %arg1, %arg2
       ret i1 %A
}

;; Note: icmp sle i32 %arg1, <immed> can always be transformed into
;;       icmp slt i32 %arg1, <immed>+1
;;
;; Consequently, even though the patterns exist to match, it's unlikely
;; they'll ever be generated.


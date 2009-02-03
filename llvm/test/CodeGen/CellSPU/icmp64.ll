; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep ceq                                %t1.s | count 20
; RUN: grep cgti                               %t1.s | count 12
; RUN: grep cgt                                %t1.s | count 16
; RUN: grep clgt                               %t1.s | count 12
; RUN: grep gb                                 %t1.s | count 12
; RUN: grep fsm                                %t1.s | count 10
; RUN: grep xori                               %t1.s | count 5
; RUN: grep selb                               %t1.s | count 18

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; $3 = %arg1, $4 = %arg2, $5 = %val1, $6 = %val2
; $3 = %arg1, $4 = %val1, $5 = %val2
;
; i64 integer comparisons:
define i64 @icmp_eq_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp eq i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_eq_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp eq i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_ne_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ne i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_ne_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ne i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_ugt_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ugt i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_ugt_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ugt i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_uge_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp uge i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_uge_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp uge i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_ult_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ult i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_ult_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ult i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_ule_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ule i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_ule_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp ule i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_sgt_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sgt i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_sgt_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sgt i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_sge_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sge i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_sge_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sge i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_slt_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp slt i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_slt_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp slt i64 %arg1, %arg2
       ret i1 %A
}

define i64 @icmp_sle_select_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sle i64 %arg1, %arg2
       %B = select i1 %A, i64 %val1, i64 %val2
       ret i64 %B
}

define i1 @icmp_sle_setcc_i64(i64 %arg1, i64 %arg2, i64 %val1, i64 %val2) nounwind {
entry:
       %A = icmp sle i64 %arg1, %arg2
       ret i1 %A
}

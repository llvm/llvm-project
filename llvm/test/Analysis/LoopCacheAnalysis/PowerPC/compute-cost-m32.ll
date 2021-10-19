; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "E-m:a-i64:64-p:32:32-n32-S128-v256:256:256-v512:512:512"
target triple = "powerpc-ibm-aix7.2.0.0"

; Check IndexedReference::computeRefCost can handle type differences between
; CacheLineSize and Numerator

; CHECK: Loop '_loop_1_do_' has cost = 2

%_elem_type_of_v = type <{ i32 }>

define signext i32 @foo(%_elem_type_of_v* %v) {
_entry:
  br label %_loop_1_do_

_loop_1_do_:                                      ; preds = %_entry, %_loop_1_do_
  %i.011 = phi i64 [ 1, %_entry ], [ %_loop_1_update_loop_ix, %_loop_1_do_ ]
  %_conv = trunc i64 %i.011 to i32
  %_ind_cast = getelementptr %_elem_type_of_v, %_elem_type_of_v* %v, i32 %_conv, i32 0
  store i32 %_conv, i32* %_ind_cast, align 4
  %_loop_1_update_loop_ix = add nuw nsw i64 %i.011, 1
  %_leq_tmp = icmp ult i64 %_loop_1_update_loop_ix, 33
  br i1 %_leq_tmp, label %_loop_1_do_, label %_loop_1_endl_

_loop_1_endl_:                                    ; preds = %_loop_1_do_
  ret i32 0
}

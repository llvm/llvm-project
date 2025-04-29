; RUN: opt -passes=loop-vectorize -disable-output -S < %s
; REQUIRES: asserts

target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

define void @__power_mod_NMOD_power_init(ptr %a, ptr %b, i32 %n) {
entry:
  br label %loop_entry

loop_exit:                                        ; preds = %loop_header
  br label %loop_entry

loop_entry:                                       ; preds = %loop_exit, %entry
  %sum.0 = phi double [ 0.000000e+00, %entry ], [ %sum.1, %loop_exit ]
  %x = load double, ptr %a, align 8
  br label %loop_header

loop_header:                                      ; preds = %loop_body, %loop_entry
  %sum.1 = phi double [ %sum.0, %loop_entry ], [ %sum.next, %loop_body ]
  %i = phi i32 [ 0, %loop_entry ], [ %i.next, %loop_body ]
  %cond = icmp sgt i32 %i, %n
  br i1 %cond, label %loop_exit, label %loop_body

loop_body:                                        ; preds = %loop_header
  store double %sum.1, ptr %b, align 8
  %sum.next = fadd reassoc double %sum.1, %x
  %i.next = add i32 %i, 1
  br label %loop_header
}


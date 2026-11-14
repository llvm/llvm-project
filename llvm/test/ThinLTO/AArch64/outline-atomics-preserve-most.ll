;; Verify that +outline-atomics-preserve-most is InlineIgnore across ThinLTO
;; module boundaries and lowers inlined atomicrmw operations using the
;; preserve-most calling convention on both Linux and Fuchsia targets (via
;; either function attributes or the backend flag).

; RUN: split-file %s %t
; RUN: sed -e "s,TRIPLE,aarch64-unknown-linux-gnu,g" %t/caller.ll | opt -module-summary -o %t/linux-caller.bc
; RUN: sed -e "s,TRIPLE,aarch64-unknown-linux-gnu,g" %t/callee.ll | opt -module-summary -o %t/linux-callee.bc
; RUN: llvm-lto2 run %t/linux-caller.bc %t/linux-callee.bc -o %t/linux-attr \
; RUN:   -r=%t/linux-caller.bc,caller,px \
; RUN:   -r=%t/linux-caller.bc,callee_atomic,l \
; RUN:   -r=%t/linux-callee.bc,callee_atomic,pl \
; RUN:   -r=%t/linux-callee.bc,callee_with_live_vals,
; RUN: llvm-objdump -d -r %t/linux-attr.1 | FileCheck %s

; RUN: llvm-lto2 run %t/linux-callee.bc -aarch64-outline-atomics-preserve-most -o %t/linux-flag \
; RUN:   -r=%t/linux-callee.bc,callee_with_live_vals,px \
; RUN:   -r=%t/linux-callee.bc,callee_atomic,pl
; RUN: llvm-objdump -d -r %t/linux-flag.1 | FileCheck %s --check-prefix=FLAG

; RUN: sed -e "s,TRIPLE,aarch64-unknown-fuchsia,g" %t/caller.ll | opt -module-summary -o %t/fuchsia-caller.bc
; RUN: sed -e "s,TRIPLE,aarch64-unknown-fuchsia,g" %t/callee.ll | opt -module-summary -o %t/fuchsia-callee.bc
; RUN: llvm-lto2 run %t/fuchsia-caller.bc %t/fuchsia-callee.bc -o %t/fuchsia-attr \
; RUN:   -r=%t/fuchsia-caller.bc,caller,px \
; RUN:   -r=%t/fuchsia-caller.bc,callee_atomic,l \
; RUN:   -r=%t/fuchsia-callee.bc,callee_atomic,pl \
; RUN:   -r=%t/fuchsia-callee.bc,callee_with_live_vals,
; RUN: llvm-objdump -d -r %t/fuchsia-attr.1 | FileCheck %s

; RUN: llvm-lto2 run %t/fuchsia-callee.bc -aarch64-outline-atomics-preserve-most -o %t/fuchsia-flag \
; RUN:   -r=%t/fuchsia-callee.bc,callee_with_live_vals,px \
; RUN:   -r=%t/fuchsia-callee.bc,callee_atomic,pl
; RUN: llvm-objdump -d -r %t/fuchsia-flag.1 | FileCheck %s --check-prefix=FLAG

; CHECK-LABEL: <caller>:
; CHECK-NOT:   stp x19
; CHECK-NOT:   stp x20
; CHECK:       bl {{.*}} <caller+{{.*}}>
; CHECK-NEXT:  R_AARCH64_CALL26 __aarch64_ldadd4_acq_rel_preservemost
; CHECK-NOT:   ldp x19
; CHECK:       ret

; FLAG-LABEL: <callee_with_live_vals>:
; FLAG-NOT:   stp x19
; FLAG-NOT:   stp x20
; FLAG:       bl {{.*}} <callee_with_live_vals+{{.*}}>
; FLAG-NEXT:  R_AARCH64_CALL26 __aarch64_ldadd4_acq_rel_preservemost
; FLAG-NOT:   ldp x19
; FLAG:       ret

;--- caller.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "TRIPLE"

declare i32 @callee_atomic(ptr, i32)

define i64 @caller(ptr %p, i32 %v, i64 %a, i64 %b, i64 %c, i64 %d) #0 {
  %old = call i32 @callee_atomic(ptr %p, i32 %v)
  %old64 = zext i32 %old to i64
  %s1 = add i64 %a, %b
  %s2 = add i64 %c, %d
  %s3 = add i64 %s1, %s2
  %res = add i64 %s3, %old64
  ret i64 %res
}

attributes #0 = { nounwind "target-features"="+outline-atomics,+outline-atomics-preserve-most" }

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "TRIPLE"

define i32 @callee_atomic(ptr %p, i32 %v) #0 {
  %old = atomicrmw add ptr %p, i32 %v acq_rel
  ret i32 %old
}

define i64 @callee_with_live_vals(ptr %p, i32 %v, i64 %a, i64 %b, i64 %c, i64 %d) #0 {
  %old = atomicrmw add ptr %p, i32 %v acq_rel
  %old64 = zext i32 %old to i64
  %s1 = add i64 %a, %b
  %s2 = add i64 %c, %d
  %s3 = add i64 %s1, %s2
  %res = add i64 %s3, %old64
  ret i64 %res
}

attributes #0 = { nounwind "target-features"="+outline-atomics" }

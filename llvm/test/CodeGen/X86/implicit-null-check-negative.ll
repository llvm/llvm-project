; RUN: llc -mtriple=x86_64-apple-macosx -O3 -debug-only=faultmaps -enable-implicit-null-checks < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; List cases where we should *not* be emitting implicit null checks.

; CHECK-NOT: Fault Map Output

define i32 @imp_null_check_load(i32* %x, i32* %y) {
 entry:
  %c = icmp eq i32* %x, null
; It isn't legal to move the load from %x from "not_null" to here --
; the store to %y could be aliasing it.
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  store i32 0, i32* %y
  %t = load i32, i32* %x
  ret i32 %t
}

define i32 @imp_null_check_gep_load(i32* %x) {
 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
; null + 5000 * sizeof(i32) lies outside the null page and hence the
; load to %t cannot be assumed to be reliably faulting.
  %x.gep = getelementptr i32, i32* %x, i32 5000
  %t = load i32, i32* %x.gep
  ret i32 %t
}

define i32 @imp_null_check_neg_gep_load(i32* %x) {
 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
; null - 5000 * sizeof(i32) lies outside the null page and hence the
; load to %t cannot be assumed to be reliably faulting.
  %x.gep = getelementptr i32, i32* %x, i32 -5000
  %t = load i32, i32* %x.gep
  ret i32 %t
}

define i32 @imp_null_check_load_no_md(i32* %x) {
; This is fine, except it is missing the !make.implicit metadata.
 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  ret i32 %t
}

define i32 @imp_null_check_no_hoist_over_acquire_load(i32* %x, i32* %y) {
; We cannot hoist %t1 over %t0 since %t0 is an acquire load
 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load atomic i32, i32* %y acquire, align 4
  %t1 = load i32, i32* %x
  %p = add i32 %t0, %t1
  ret i32 %p
}

define i32 @imp_null_check_add_result(i32* %x, i32* %y) {
; This will codegen to:
;
;   movl    (%rsi), %eax
;   addl    (%rdi), %eax
;
; The load instruction we wish to hoist is the addl, but there is a
; write-after-write hazard preventing that from happening.  We could
; get fancy here and exploit the commutativity of addition, but right
; now -implicit-null-checks isn't that smart.
;

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load i32, i32* %y
  %t1 = load i32, i32* %x
  %p = add i32 %t0, %t1
  ret i32 %p
}

!0 = !{}

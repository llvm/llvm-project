; RUN: llc -mtriple=x86_64-apple-macosx -O3 -debug-only=faultmaps -enable-implicit-null-checks < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; List cases where we should *not* be emitting implicit null checks.

; CHECK-NOT: Fault Map Output

define i32 @imp_null_check_load(ptr %x, ptr %y) {
 entry:
  %c = icmp eq ptr %x, null
; It isn't legal to move the load from %x from "not_null" to here --
; the store to %y could be aliasing it.
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  store i32 0, ptr %y
  %t = load i32, ptr %x
  ret i32 %t
}

define i32 @imp_null_check_gep_load(ptr %x) {
 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
; null + 5000 * sizeof(i32) lies outside the null page and hence the
; load to %t cannot be assumed to be reliably faulting.
  %x.gep = getelementptr i32, ptr %x, i32 5000
  %t = load i32, ptr %x.gep
  ret i32 %t
}

define i32 @imp_null_check_neg_gep_load(ptr %x) {
 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
; null - 5000 * sizeof(i32) lies outside the null page and hence the
; load to %t cannot be assumed to be reliably faulting.
  %x.gep = getelementptr i32, ptr %x, i32 -5000
  %t = load i32, ptr %x.gep
  ret i32 %t
}

define i32 @imp_null_check_load_no_md(ptr %x) {
; This is fine, except it is missing the !make.implicit metadata.
 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  ret i32 %t
}

define i32 @imp_null_check_no_hoist_over_acquire_load(ptr %x, ptr %y) {
; We cannot hoist %t1 over %t0 since %t0 is an acquire load
 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load atomic i32, ptr %y acquire, align 4
  %t1 = load i32, ptr %x
  %p = add i32 %t0, %t1
  ret i32 %p
}

define i32 @imp_null_check_add_result(ptr %x, ptr %y) {
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
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load i32, ptr %y
  %t1 = load i32, ptr %x
  %p = add i32 %t0, %t1
  ret i32 %p
}

; This redefines the null check reg by doing a zero-extend, a shift on
; itself and then an add.
; Cannot be converted to implicit check since the zero reg is no longer zero.
define i64 @imp_null_check_load_shift_add_addr(ptr %x, i64 %r) {
  entry:
   %c = icmp eq ptr %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint ptr %x to i64
   %shry = shl i64 %y, 6
   %shry.add = add i64 %shry, %r
   %y.ptr = inttoptr i64 %shry.add to ptr
   %x.loc = getelementptr i64, ptr %y.ptr, i64 1
   %t = load i64, ptr %x.loc
   ret i64 %t
}

; the memory op is not within faulting page.
define i64 @imp_null_check_load_addr_outside_faulting_page(ptr %x) {
  entry:
   %c = icmp eq ptr %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint ptr %x to i64
   %shry = shl i64 %y, 3
   %shry.add = add i64 %shry, 68719472640
   %y.ptr = inttoptr i64 %shry.add to ptr
   %x.loc = getelementptr i64, ptr %y.ptr, i64 1
   %t = load i64, ptr %x.loc
   ret i64 %t
}

!0 = !{}

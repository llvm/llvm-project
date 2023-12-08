;; Sparc backend can't currently handle variable allocas with
;; alignment greater than the stack alignment.  This code ought to
;; compile, but doesn't currently.

;; RUN: not --crash llc -march=sparc < %s 2>&1 | FileCheck %s
;; RUN: not --crash llc -march=sparcv9 < %s 2>&1 | FileCheck %s
;; CHECK: ERROR: Function {{.*}} required stack re-alignment

define void @variable_alloca_with_overalignment(i32 %num) {
  %aligned = alloca i32, align 64
  %var_size = alloca i8, i32 %num, align 4
  call void @foo(i32* %aligned, i8* %var_size)
  ret void
}

;; Same but with the alloca itself overaligned
define void @variable_alloca_with_overalignment_2(i32 %num) {
  %var_size = alloca i8, i32 %num, align 64
  call void @foo(i32* null, i8* %var_size)
  ret void
}

declare void @foo(i32*, i8*);

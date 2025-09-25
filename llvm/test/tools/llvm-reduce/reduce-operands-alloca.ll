; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s --check-prefixes=CHECK,ZERO < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s --check-prefixes=CHECK,ONE < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-poison --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s --check-prefixes=CHECK,POISON < %t


; CHECK-LABEL: @dyn_alloca(
; ZERO: %alloca = alloca i32, i32 %size, align 4
; ONE: %alloca = alloca i32, align 4
; POISON: %alloca = alloca i32, i32 %size, align 4
define void @dyn_alloca(i32 %size) {
 %alloca = alloca i32, i32 %size
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_0_elt(
; ZERO: %alloca = alloca i32, i32 0, align 4
; ONE: %alloca = alloca i32, i32 0, align 4
; POISON:  %alloca = alloca i32, i32 0, align 4
define void @alloca_0_elt() {
 %alloca = alloca i32, i32 0
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_1_elt(
; ZERO: %alloca = alloca i32, align 4
; ONE: %alloca = alloca i32, align 4
; POISON: %alloca = alloca i32, align 4
define void @alloca_1_elt() {
 %alloca = alloca i32, i32 1
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_1024_elt(
; ZERO: %alloca = alloca i32, i32 1024, align 4
; ONE: %alloca = alloca i32, align 4
; POISON: %alloca = alloca i32, i32 1024, align 4
define void @alloca_1024_elt() {
 %alloca = alloca i32, i32 1024
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_poison_elt(
; ZERO: %alloca = alloca i32, i32 poison, align 4
; ONE: %alloca = alloca i32, align 4
; POISON: %alloca = alloca i32, i32 poison, align 4
define void @alloca_poison_elt() {
 %alloca = alloca i32, i32 poison
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_constexpr_elt(
; ZERO: %alloca = alloca i32, i32 ptrtoint (ptr @alloca_constexpr_elt to i32)
; ONE: %alloca = alloca i32, align 4
; POISON: %alloca = alloca i32, i32 ptrtoint (ptr @alloca_constexpr_elt to i32)
define void @alloca_constexpr_elt() {
 %alloca = alloca i32, i32 ptrtoint (ptr @alloca_constexpr_elt to i32)
 store i32 0, ptr %alloca
 ret void
}

; CHECK-LABEL: @alloca_lifetimes(
; ZERO: call void @llvm.lifetime.start.p0(ptr %alloca)
; ONE: call void @llvm.lifetime.start.p0(ptr %alloca)
; POISON: call void @llvm.lifetime.start.p0(ptr %alloca)
define void @alloca_lifetimes() {
  %alloca = alloca i32
  call void @llvm.lifetime.start.p0(ptr %alloca)
  store i32 0, ptr %alloca
  call void @llvm.lifetime.end.p0(ptr %alloca)
  ret void
}

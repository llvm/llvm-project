; RUN: llc < %s -mtriple=sparc -mattr=hard-quad-float | FileCheck %s --check-prefix=CHECK --check-prefix=HARD --check-prefix=BE
; RUN: llc < %s -mtriple=sparcel -mattr=hard-quad-float | FileCheck %s --check-prefix=CHECK --check-prefix=HARD --check-prefix=EL
; RUN: llc < %s -mtriple=sparc -mattr=-hard-quad-float -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT --check-prefix=BE
; RUN: llc < %s -mtriple=sparcel -mattr=-hard-quad-float | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT --check-prefix=EL

; CHECK-LABEL: f128_ops:
; CHECK:      ldd
; CHECK:      ldd
; CHECK:      ldd
; CHECK:      ldd
; HARD:       faddq [[R0:.+]],  [[R1:.+]],  [[R2:.+]]
; HARD:       fsubq [[R2]], [[R3:.+]], [[R4:.+]]
; HARD:       fmulq [[R4]], [[R5:.+]], [[R6:.+]]
; HARD:       fdivq [[R6]], [[R2]]
; SOFT:       call _Q_add
; SOFT:       unimp 16
; SOFT:       call _Q_sub
; SOFT:       unimp 16
; SOFT:       call _Q_mul
; SOFT:       unimp 16
; SOFT:       call _Q_div
; SOFT:       unimp 16
; CHECK:      std
; CHECK:      std

define void @f128_ops(ptr noalias sret(fp128) %scalar.result, ptr byval(fp128) %a, ptr byval(fp128) %b, ptr byval(fp128) %c, ptr byval(fp128) %d) {
entry:
  %0 = load fp128, ptr %a, align 8
  %1 = load fp128, ptr %b, align 8
  %2 = load fp128, ptr %c, align 8
  %3 = load fp128, ptr %d, align 8
  %4 = fadd fp128 %0, %1
  %5 = fsub fp128 %4, %2
  %6 = fmul fp128 %5, %3
  %7 = fdiv fp128 %6, %4
  store fp128 %7, ptr %scalar.result, align 8
  ret void
}

; CHECK-LABEL: f128_spill:
; CHECK:       std %f{{.+}}, [%[[S0:.+]]]
; CHECK:       std %f{{.+}}, [%[[S1:.+]]]
; CHECK-DAG:   ldd [%[[S0]]], %f{{.+}}
; CHECK-DAG:   ldd [%[[S1]]], %f{{.+}}
; CHECK:       jmp {{%[oi]7}}+12

define void @f128_spill(ptr noalias sret(fp128) %scalar.result, ptr byval(fp128) %a) {
entry:
  %0 = load fp128, ptr %a, align 8
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  store fp128 %0, ptr %scalar.result, align 8
  ret void
}

; CHECK-LABEL: f128_spill_large:
; CHECK:       sethi 4, %g1

; CHECK:       std %f{{.+}}, [%[[S0:.+]]]
; CHECK:       std %f{{.+}}, [%[[S1:.+]]]
; CHECK-DAG:   ldd [%[[S0]]], %f{{.+}}
; CHECK-DAG:   ldd [%[[S1]]], %f{{.+}}
define void @f128_spill_large(ptr noalias sret(<251 x fp128>) %scalar.result, ptr byval(<251 x fp128>) %a) {
entry:
  %0 = load <251 x fp128>, ptr %a, align 8
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  store <251 x fp128> %0, ptr %scalar.result, align 8
  ret void
}

; CHECK-LABEL: f128_compare:
; HARD:       fcmpq
; HARD-NEXT:  nop
; SOFT:       _Q_cmp

define i32 @f128_compare(ptr byval(fp128) %f0, ptr byval(fp128) %f1, i32 %a, i32 %b) {
entry:
   %0 = load fp128, ptr %f0, align 8
   %1 = load fp128, ptr %f1, align 8
   %cond = fcmp ult fp128 %0, %1
   %ret = select i1 %cond, i32 %a, i32 %b
   ret i32 %ret
}

; CHECK-LABEL: f128_compare2:
; HARD:        fcmpq
; HARD:        fb{{ule|g}}
; SOFT:       _Q_cmp
; SOFT:       cmp

define i32 @f128_compare2(ptr byval(fp128) %f0) {
entry:
  %0 = load fp128, ptr %f0, align 8
  %1 = fcmp ogt fp128 %0, 0xL00000000000000000000000000000000
  br i1 %1, label %"5", label %"7"

"5":                                              ; preds = %entry
  ret i32 0

"7":                                              ; preds = %entry
  ret i32 1
}


; CHECK-LABEL: f128_abs:
; CHECK-DAG:       ldd [%o0], [[REG:%f[0-9]+]]
; CHECK-DAG:       ldd [%o0+8], %f{{[0-9]+}}
; BE:          fabss [[REG]], [[REG]]
; EL:          fabss %f1, %f1

define void @f128_abs(ptr noalias sret(fp128) %scalar.result, ptr byval(fp128) %a) {
entry:
  %0 = load fp128, ptr %a, align 8
  %1 = tail call fp128 @llvm.fabs.f128(fp128 %0)
  store fp128 %1, ptr %scalar.result, align 8
  ret void
}

declare fp128 @llvm.fabs.f128(fp128) nounwind readonly

; CHECK-LABEL: int_to_f128:
; HARD:       fitoq
; SOFT:       _Q_itoq
; SOFT:       unimp 16

define void @int_to_f128(ptr noalias sret(fp128) %scalar.result, i32 %i) {
entry:
  %0 = sitofp i32 %i to fp128
  store fp128 %0, ptr %scalar.result, align 8
  ret void
}

; CHECK-LABEL: fp128_unaligned:
; CHECK:       ldub
; HARD:        faddq
; SOFT:       call _Q_add
; SOFT:       unimp 16
; CHECK:       stb
; CHECK:       ret

define void @fp128_unaligned(ptr %a, ptr %b, ptr %c) {
entry:
  %0 = load fp128, ptr %a, align 1
  %1 = load fp128, ptr %b, align 1
  %2 = fadd fp128 %0, %1
  store fp128 %2, ptr %c, align 1
  ret void
}

; CHECK-LABEL: uint_to_f128:
; HARD:       fdtoq
; SOFT:       _Q_utoq
; SOFT:       unimp 16

define void @uint_to_f128(ptr noalias sret(fp128) %scalar.result, i32 %i) {
entry:
  %0 = uitofp i32 %i to fp128
  store fp128 %0, ptr %scalar.result, align 8
  ret void
}

; CHECK-LABEL: f128_to_i32:
; HARD:       fqtoi
; HARD:       fqtoi
; SOFT:       call _Q_qtou
; SOFT:       call _Q_qtoi


define i32 @f128_to_i32(ptr %a, ptr %b) {
entry:
  %0 = load fp128, ptr %a, align 8
  %1 = load fp128, ptr %b, align 8
  %2 = fptoui fp128 %0 to i32
  %3 = fptosi fp128 %1 to i32
  %4 = add i32 %2, %3
  ret i32 %4
}

; CHECK-LABEL:   test_itoq_qtoi
; HARD-DAG:      call _Q_lltoq
; HARD-DAG:      call _Q_qtoll
; HARD-DAG:      fitoq
; HARD-DAG:      fqtoi
; SOFT-DAG:      call _Q_lltoq
; SOFT-DAG:      unimp 16
; SOFT-DAG:      call _Q_qtoll
; SOFT-DAG:      call _Q_itoq
; SOFT-DAG:      unimp 16
; SOFT-DAG:      call _Q_qtoi

define void @test_itoq_qtoi(i64 %a, i32 %b, ptr %c, ptr %d, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = sitofp i64 %a to fp128
  store  fp128 %0, ptr %ptr1, align 8
  %cval = load fp128, ptr %c, align 8
  %1 = fptosi fp128 %cval to i64
  store  i64 %1, ptr %ptr0, align 8
  %2 = sitofp i32 %b to fp128
  store  fp128 %2, ptr %ptr1, align 8
  %dval = load fp128, ptr %d, align 8
  %3 = fptosi fp128 %dval to i32
  %4 = bitcast ptr %ptr0 to ptr
  store  i32 %3, ptr %4, align 8
  ret void
}

; CHECK-LABEL:   test_utoq_qtou:
; CHECK-DAG:     call _Q_ulltoq
; CHECK-DAG:     call _Q_qtoull
; HARD-DAG:      fdtoq
; HARD-DAG:      fqtoi
; SOFT-DAG:      call _Q_utoq
; SOFT-DAG:      unimp 16
; SOFT-DAG:      call _Q_qtou

define void @test_utoq_qtou(i64 %a, i32 %b, ptr %c, ptr %d, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = uitofp i64 %a to fp128
  store  fp128 %0, ptr %ptr1, align 8
  %cval = load fp128, ptr %c, align 8
  %1 = fptoui fp128 %cval to i64
  store  i64 %1, ptr %ptr0, align 8
  %2 = uitofp i32 %b to fp128
  store  fp128 %2, ptr %ptr1, align 8
  %dval = load fp128, ptr %d, align 8
  %3 = fptoui fp128 %dval to i32
  %4 = bitcast ptr %ptr0 to ptr
  store  i32 %3, ptr %4, align 8
  ret void
}

; CHECK-LABEL: f128_neg:
; CHECK-DAG:       ldd [%o0], [[REG:%f[0-9]+]]
; CHECK-DAG:       ldd [%o0+8], %f{{[0-9]+}}
; BE:          fnegs [[REG]], [[REG]]
; LE:          fnegs [[REG]], [[REG]]

define void @f128_neg(ptr noalias sret(fp128) %scalar.result, ptr byval(fp128) %a) {
entry:
  %0 = load fp128, ptr %a, align 8
  %1 = fsub fp128 0xL00000000000000008000000000000000, %0
  store fp128 %1, ptr %scalar.result, align 8
  ret void
}

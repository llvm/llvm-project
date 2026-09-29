; RUN: llc -mtriple=sparc < %s | FileCheck %s -check-prefix=V8 -check-prefix=V8-BE
; RUN: llc -mtriple=sparcel < %s | FileCheck %s -check-prefix=V8 -check-prefix=V8-EL
; RUN: llc -mtriple=sparc -O0 < %s | FileCheck %s -check-prefix=V8-UNOPT
; RUN: llc -mtriple=sparc -mattr=v9 < %s | FileCheck %s -check-prefix=V9
; RUN: llc -mtriple=sparc64-unknown-linux < %s | FileCheck %s -check-prefix=SPARC64

; V8-LABEL:     test_neg:
; V8:     call get_double
; V8-BE:     fnegs %f0, %f0
; V8-EL:     fnegs %f1, %f1

; V8-UNOPT-LABEL:     test_neg:
; V8-UNOPT:     fnegs
; V8-UNOPT:     ! implicit-def
; V8-UNOPT:     fmovs {{.+}}, %f0
; V8-UNOPT:     fmovs {{.+}}, %f1

; V9-LABEL:     test_neg:
; V9:     fnegd %f0, %f0

; SPARC64-LABEL: test_neg:
; SPARC64:       fnegd %f0, %f0

define double @test_neg() {
entry:
  %0 = tail call double @get_double()
  %1 = fsub double -0.000000e+00, %0
  ret double %1
}

; V8-LABEL:     test_abs:
; V8-BE:     fabss %f0, %f0
; V8-EL:     fabss %f1, %f1

; V8-UNOPT-LABEL:     test_abs:
; V8-UNOPT:     fabss
; V8-UNOPT:     ! implicit-def
; V8-UNOPT:     fmovs {{.+}}, %f0
; V8-UNOPT:     fmovs {{.+}}, %f1

; V9-LABEL:     test_abs:
; V9:     fabsd %f0, %f0


; SPARC64-LABEL:     test_abs:
; SPARC64:     fabsd %f0, %f0

define double @test_abs() {
entry:
  %0 = tail call double @get_double()
  %1 = tail call double @llvm.fabs.f64(double %0)
  ret double %1
}

declare double @get_double()
declare double @llvm.fabs.f64(double) nounwind readonly

; V8-LABEL:    test_v9_floatreg:
; V8:          fsubd {{.+}}, {{.+}}, [[R:%f(((1|2)?(0|2|4|6|8))|30)]]
; V8:          std [[R]], [%{{.+}}]
; V8:          ldd [%{{.+}}], %f0
; V8:          faddd {{.+}}, {{.+}}, {{.+}}

; V9-LABEL:    test_v9_floatreg:
; V9:          fsubd {{.+}}, {{.+}}, {{.+}}
; V9:          faddd {{.+}}, {{.+}}, %f0

; SPARC64-LABEL:    test_v9_floatreg:
; SPARC64:          fsubd {{.+}}, {{.+}}, {{.+}}
; SPARC64:          faddd {{.+}}, {{.+}}, %f0

define double @test_v9_floatreg() {
entry:
  %0 = tail call double @get_double()
  %1 = tail call double @get_double()
  %2 = fsub double %0, %1
  tail call void asm sideeffect "", "~{f0},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %3 = fadd double %2, %2
  ret double %3
}

; V8-LABEL:    test_xtos_stox
; V8:          call __floatdisf
; V8:          call __fixsfdi

; V9-LABEL:    test_xtos_stox
; V9:          call __floatdisf
; V9:          call __fixsfdi

; SPARC64-LABEL:    test_xtos_stox
; SPARC64:          fxtos
; SPARC64:          fstox

define void @test_xtos_stox(i64 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = sitofp i64 %a to float
  store float %0, ptr %ptr1, align 8
  %1 = fptosi float %0 to i64
  store i64 %1, ptr %ptr0, align 8
  ret void
}

; V8-LABEL:    test_itos_stoi
; V8:          fitos
; V8:          fstoi

; V9-LABEL:    test_itos_stoi
; V9:          fitos
; V9:          fstoi

; SPARC64-LABEL:    test_itos_stoi
; SPARC64:          fitos
; SPARC64:          fstoi

define void @test_itos_stoi(i32 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = sitofp i32 %a to float
  store float %0, ptr %ptr1, align 8
  %1 = fptosi float %0 to i32
  store i32 %1, ptr %ptr0, align 8
  ret void
}


; V8-LABEL:    test_xtod_dtox
; V8:          call __floatdidf
; V8:          call __fixdfdi

; V9-LABEL:    test_xtod_dtox
; V9:          call __floatdidf
; V9:          call __fixdfdi

; SPARC64-LABEL:    test_xtod_dtox
; SPARC64:          fxtod
; SPARC64:          fdtox

define void @test_xtod_dtox(i64 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = sitofp i64 %a to double
  store double %0, ptr %ptr1, align 8
  %1 = fptosi double %0 to i64
  store i64 %1, ptr %ptr0, align 8
  ret void
}

; V8-LABEL:    test_itod_dtoi
; V8:          fitod
; V8:          fdtoi

; V9-LABEL:    test_itod_dtoi
; V9:          fitod
; V9:          fdtoi

; SPARC64-LABEL:    test_itod_dtoi
; SPARC64:          fitod
; SPARC64:          fdtoi

define void @test_itod_dtoi(i32 %a, double %b, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = sitofp i32 %a to double
  store double %0, ptr %ptr1, align 8
  %1 = fptosi double %b to i32
  store i32 %1, ptr %ptr0, align 8
  ret void
}

; V8-LABEL:    test_uxtos_stoux
; V8:          call __floatundisf
; V8:          call __fixunssfdi

; V9-LABEL:    test_uxtos_stoux
; V9:          call __floatundisf
; V9:          call __fixunssfdi

; SPARC64-LABEL:   test_uxtos_stoux
; SPARC64-NOT:     call __floatundisf
; SPARC64-NOT:     call __fixunssfdi

define void @test_uxtos_stoux(i64 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = uitofp i64 %a to float
  store float %0, ptr %ptr1, align 8
  %1 = fptoui float %0 to i64
  store i64 %1, ptr %ptr0, align 8
  ret void
}

; V8-LABEL:    test_utos_stou
; V8:          fdtos
; V8:          fstoi

; V9-LABEL:    test_utos_stou
; V9:          fdtos
; V9:          fstoi

; SPARC64-LABEL:    test_utos_stou
; SPARC64:     fdtos
; SPARC64:     fstoi

define void @test_utos_stou(i32 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = uitofp i32 %a to float
  store float %0, ptr %ptr1, align 8
  %1 = fptoui float %0 to i32
  store i32 %1, ptr %ptr0, align 8
  ret void
}


; V8-LABEL:    test_uxtod_dtoux
; V8:          call __floatundidf
; V8:          call __fixunsdfdi

; V9-LABEL:    test_uxtod_dtoux
; V9:          call __floatundidf
; V9:          call __fixunsdfdi

; SPARC64-LABEL:    test_uxtod_dtoux
; SPARC64-NOT:          call __floatundidf
; SPARC64-NOT:          call __floatunsdfdi

define void @test_uxtod_dtoux(i64 %a, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = uitofp i64 %a to double
  store double %0, ptr %ptr1, align 8
  %1 = fptoui double %0 to i64
  store i64 %1, ptr %ptr0, align 8
  ret void
}

; V8-LABEL:    test_utod_dtou
; V8-NOT:      fitod
; V8:          fdtoi

; V9-LABEL:    test_utod_dtou
; V9-NOT:      fitod
; V9:          fdtoi

; SPARC64-LABEL:    test_utod_dtou
; SPARC64-NOT:      fitod
; SPARC64:          fdtoi

define void @test_utod_dtou(i32 %a, double %b, ptr %ptr0, ptr %ptr1) {
entry:
  %0 = uitofp i32 %a to double
  store double %0, ptr %ptr1, align 8
  %1 = fptoui double %b to i32
  store i32 %1, ptr %ptr0, align 8
  ret void
}

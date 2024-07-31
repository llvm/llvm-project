; RUN: llc -march=sparc <%s | FileCheck %s

; CHECK-LABEL: test_constraint_r
; CHECK:       add %i1, %i0, %i0
define i32 @test_constraint_r(i32 %a, i32 %b) {
entry:
  %0 = tail call i32 asm sideeffect "add $2, $1, $0", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %0
}

;; Check tests only that the constraints are accepted without a compiler failure.
; CHECK-LABEL: test_constraints_nro:
%struct.anon = type { i32, i32 }
@v = external global %struct.anon, align 4
define void @test_constraints_nro() {
entry:
  %0 = load i32, ptr @v;
  %1 = load i32, ptr getelementptr inbounds (%struct.anon, ptr @v, i32 0, i32 1);
  tail call void asm sideeffect "", "nro,nro"(i32 %0, i32 %1)
  ret void
}

; CHECK-LABEL: test_constraint_I:
; CHECK:       add %i0, 1023, %i0
define i32 @test_constraint_I(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 1023)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_I_neg:
; CHECK:       add %i0, -4096, %i0
define i32 @test_constraint_I_neg(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 -4096)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_I_largeimm:
; CHECK:       sethi 9, [[R0:%[gilo][0-7]]]
; CHECK:       or [[R0]], 784, [[R1:%[gilo][0-7]]]
; CHECK:       add %i0, [[R1]], %i0
define i32 @test_constraint_I_largeimm(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 10000)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_reg:
; CHECK:       ldda [%i1] 43, %g2
; CHECK:       ldda [%i1] 43, %g4
define void @test_constraint_reg(i32 %s, ptr %ptr) {
entry:
  %0 = tail call i64 asm sideeffect "ldda [$1] $2, $0", "={r2},r,n"(ptr %ptr, i32 43)
  %1 = tail call i64 asm sideeffect "ldda [$1] $2, $0", "={g4},r,n"(ptr %ptr, i32 43)
  ret void
}

;; Ensure that i64 args to asm are allocated to the IntPair register class.
;; Also checks that there's no register renaming for leaf proc if it has inline asm.
; CHECK-LABEL: test_constraint_r_i64:
; CHECK: mov %i0, %i5
; CHECK: sra %i5, 31, %i4
; CHECK: std %i4, [%i1]
define i32 @test_constraint_r_i64(i32 %foo, ptr %out, i32 %o) {
entry:
  %conv = sext i32 %foo to i64
  tail call void asm sideeffect "std $0, [$1]", "r,r,~{memory}"(i64 %conv, ptr %out)
  ret i32 %o
}

;; Same test without leaf-proc opt
; CHECK-LABEL: test_constraint_r_i64_noleaf:
; CHECK: mov %i0, %i5
; CHECK: sra %i5, 31, %i4
; CHECK: std %i4, [%i1]
define i32 @test_constraint_r_i64_noleaf(i32 %foo, ptr %out, i32 %o) #0 {
entry:
  %conv = sext i32 %foo to i64
  tail call void asm sideeffect "std $0, [$1]", "r,r,~{memory}"(i64 %conv, ptr %out)
  ret i32 %o
}
attributes #0 = { "frame-pointer"="all" }

;; Ensures that tied in and out gets allocated properly.
; CHECK-LABEL: test_i64_inout:
; CHECK: mov 5, %i3
; CHECK: mov %g0, %i2
; CHECK: xor %i2, %g0, %i2
; CHECK: mov %i2, %i0
; CHECK: ret
define i64 @test_i64_inout() {
entry:
  %0 = call i64 asm sideeffect "xor $1, %g0, $0", "=r,0,~{i1}"(i64 5);
  ret i64 %0
}


;; Ensures that inline-asm accepts and uses 'f' and 'e' register constraints.
; CHECK-LABEL: fadds:
; CHECK: fadds  %f0, %f1, %f0
define float @fadds(float, float) local_unnamed_addr #2 {
entry:
  %2 = tail call float asm sideeffect "fadds  $1, $2, $0;", "=f,f,e"(float %0, float %1) #7
  ret float %2
}

; CHECK-LABEL: faddd:
; CHECK: faddd  %f0, %f2, %f0
define double @faddd(double, double) local_unnamed_addr #2 {
entry:
  %2 = tail call double asm sideeffect "faddd  $1, $2, $0;", "=f,f,e"(double %0, double %1) #7
  ret double %2
}

; CHECK-LABEL: test_addressing_mode_i64:
; CHECK: std %l0, [%i0]
define void @test_addressing_mode_i64(ptr %out) {
entry:
  call void asm "std %l0, $0", "=*m,r"(ptr elementtype(i64) nonnull %out, i64 0)
  ret void
}

; CHECK-LABEL: test_constraint_float_reg:
; CHECK: fadds %f20, %f20, %f20
; CHECK: faddd %f20, %f20, %f20
define void @test_constraint_float_reg() {
entry:
  tail call void asm sideeffect "fadds $0,$1,$2", "{f20},{f20},{f20}"(float 6.0, float 7.0, float 8.0)
  tail call void asm sideeffect "faddd $0,$1,$2", "{f20},{f20},{f20}"(double 9.0, double 10.0, double 11.0)
  ret void
}

; CHECK-LABEL: test_constraint_f_e_i32_i64:
; CHECK: ld [%i0+%lo(.LCPI13_0)], %f0
; CHECK: ldd [%i0+%lo(.LCPI13_1)], %f2
; CHECK: fadds %f0, %f0, %f0
; CHECK: faddd %f2, %f2, %f0

define void @test_constraint_f_e_i32_i64() {
entry:
  %0 = call float asm sideeffect "fadds $1, $2, $0", "=f,f,e"(i32 0, i32 0)
  %1 = call double asm sideeffect "faddd $1, $2, $0", "=f,f,e"(i64 0, i64 0)
  ret void
}

; CHECK-LABEL: test_twinword:
; CHECK: rd  %asr5, %i1
; CHECK: srlx %i1, 32, %i0

define i64 @test_twinword(){
  %1 = tail call i64 asm sideeffect "rd %asr5, ${0:L} \0A\09 srlx ${0:L}, 32, ${0:H}", "={i0}"()
  ret i64 %1
}

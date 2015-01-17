; RUN: llc -march=sparc <%s | FileCheck %s

; CHECK-LABEL: test_constraint_r
; CHECK:       add %o1, %o0, %o0
define i32 @test_constraint_r(i32 %a, i32 %b) {
entry:
  %0 = tail call i32 asm sideeffect "add $2, $1, $0", "=r,r,r"(i32 %a, i32 %b)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_I
; CHECK:       add %o0, 1023, %o0
define i32 @test_constraint_I(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 1023)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_I_neg
; CHECK:       add %o0, -4096, %o0
define i32 @test_constraint_I_neg(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 -4096)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_I_largeimm
; CHECK:       sethi 9, [[R0:%[gilo][0-7]]]
; CHECK:       or [[R0]], 784, [[R1:%[gilo][0-7]]]
; CHECK:       add %o0, [[R1]], %o0
define i32 @test_constraint_I_largeimm(i32 %a) {
entry:
  %0 = tail call i32 asm sideeffect "add $1, $2, $0", "=r,r,rI"(i32 %a, i32 10000)
  ret i32 %0
}

; CHECK-LABEL: test_constraint_reg
; CHECK:       ldda [%o1] 43, %g2
; CHECK:       ldda [%o1] 43, %g3
define void @test_constraint_reg(i32 %s, i32* %ptr) {
entry:
  %0 = tail call i64 asm sideeffect "ldda [$1] $2, $0", "={r2},r,n"(i32* %ptr, i32 43)
  %1 = tail call i64 asm sideeffect "ldda [$1] $2, $0", "={g3},r,n"(i32* %ptr, i32 43)
  ret void
}

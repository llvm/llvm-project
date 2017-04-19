; RUN: llc -mtriple=thumbv7k-apple-watchos2.0 -arm-atomic-cfg-tidy=0 -o - %s | FileCheck %s

@tls_var = thread_local global i32 0

; r9 and r12 can be live across the asm, but those get clobbered by the TLS
; access (in a different BB to order it).
define i32 @test_regs_preserved(i32* %ptr1, i32* %ptr2, i1 %tst1) {
; CHECK-LABEL: test_regs_preserved:
; CHECK: str {{.*}}, [sp
; CHECK: mov {{.*}}, r12
entry:
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r10},~{r11},~{r13},~{lr}"()
  br i1 %tst1, label %get_tls, label %done

get_tls:
  %val = load i32, i32* @tls_var
  br label %done

done:
  %res = phi i32 [%val, %get_tls], [0, %entry]
  store i32 42, i32* %ptr1
  store i32 42, i32* %ptr2
  ret i32 %res
}

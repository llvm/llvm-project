; RUN: llc -O0 -disable-fp-elim -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s -check-prefix=PPC32
; RUN: llc -O0 -disable-fp-elim -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefix=PPC64

declare void @foo()

define i32 @test_cr2() nounwind {
entry:
  %ret = alloca i32, align 4
  %0 = call i32 asm sideeffect "\0A\09mtcr $4\0A\09cmp 2,$2,$1\0A\09mfcr $0", "=r,r,r,r,r,~{cr2}"(i32 1, i32 2, i32 3, i32 0) nounwind
  store i32 %0, i32* %ret, align 4
  call void @foo()
  %1 = load i32* %ret, align 4
  ret i32 %1
}

; PPC32: mfcr 12
; PPC32-NEXT: stw 12, {{[0-9]+}}(31)
; PPC32: lwz 12, {{[0-9]+}}(31)
; PPC32-NEXT: mtcrf 32, 12

; PPC64: mfcr 12
; PPC64-NEXT: stw 12, 8(1)
; PPC64: lwz 12, 8(1)
; PPC64-NEXT: mtcrf 32, 12

define i32 @test_cr234() nounwind {
entry:
  %ret = alloca i32, align 4
  %0 = call i32 asm sideeffect "\0A\09mtcr $4\0A\09cmp 2,$2,$1\0A\09cmp 3,$2,$2\0A\09cmp 4,$2,$3\0A\09mfcr $0", "=r,r,r,r,r,~{cr2},~{cr3},~{cr4}"(i32 1, i32 2, i32 3, i32 0) nounwind
  store i32 %0, i32* %ret, align 4
  call void @foo()
  %1 = load i32* %ret, align 4
  ret i32 %1
}

; PPC32: mfcr 12
; PPC32-NEXT: stw 12, {{[0-9]+}}(31)
; PPC32: lwz 12, {{[0-9]+}}(31)
; PPC32-NEXT: mtcrf 32, 12
; PPC32-NEXT: mtcrf 16, 12
; PPC32-NEXT: mtcrf 8, 12

; PPC64: mfcr 12
; PPC64-NEXT: stw 12, 8(1)
; PPC64: lwz 12, 8(1)
; PPC64-NEXT: mtcrf 32, 12
; PPC64-NEXT: mtcrf 16, 12
; PPC64-NEXT: mtcrf 8, 12


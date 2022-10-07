; RUN: llc -O0 -frame-pointer=all -mtriple=powerpc-unknown-linux-gnu -mcpu=g5 < %s | FileCheck %s -check-prefix=PPC32
; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 < %s | FileCheck %s -check-prefix=PPC64
; RUN: llc -O0 -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s -check-prefix=PPC64-ELFv2

declare void @foo()

define i32 @test_cr2() nounwind uwtable {
entry:
  %ret = alloca i32, align 4
  %0 = call i32 asm sideeffect "\0A\09mtcr $4\0A\09cmpw 2,$2,$1\0A\09mfcr $0", "=r,r,r,r,r,~{cr2}"(i32 1, i32 2, i32 3, i32 0) nounwind
  store i32 %0, ptr %ret, align 4
  call void @foo()
  %1 = load i32, ptr %ret, align 4
  ret i32 %1
}

; PPC32-LABEL: test_cr2:
; PPC32: stwu 1, -32(1)
; PPC32: stw 31, 28(1)
; PPC32: mfcr 12
; PPC32-NEXT: stw 12, 24(31)
; PPC32: lwz 12, 24(31)
; PPC32-NEXT: mtocrf 32, 12

; PPC64: .cfi_startproc
; PPC64: mfcr 12
; PPC64: stw 12, 8(1)
; PPC64: stdu 1, -[[AMT:[0-9]+]](1)
; PPC64: .cfi_def_cfa_offset 128
; PPC64: .cfi_offset lr, 16
; PPC64: .cfi_offset cr2, 8
; PPC64: addi 1, 1, [[AMT]]
; PPC64: lwz 12, 8(1)
; PPC64: mtocrf 32, 12
; PPC64: .cfi_endproc

define i32 @test_cr234() nounwind {
entry:
  %ret = alloca i32, align 4
  %0 = call i32 asm sideeffect "\0A\09mtcr $4\0A\09cmpw 2,$2,$1\0A\09cmpw 3,$2,$2\0A\09cmpw 4,$2,$3\0A\09mfcr $0", "=r,r,r,r,r,~{cr2},~{cr3},~{cr4}"(i32 1, i32 2, i32 3, i32 0) nounwind
  store i32 %0, ptr %ret, align 4
  call void @foo()
  %1 = load i32, ptr %ret, align 4
  ret i32 %1
}

; PPC32-LABEL: test_cr234:
; PPC32: stwu 1, -32(1)
; PPC32: stw 31, 28(1)
; PPC32: mfcr 12
; PPC32-NEXT: stw 12, 24(31)
; PPC32: lwz 12, 24(31)
; PPC32-NEXT: mtocrf 32, 12
; PPC32-NEXT: mtocrf 16, 12
; PPC32-NEXT: mtocrf 8, 12

; PPC64: mfcr 12
; PPC64: stw 12, 8(1)
; PPC64: stdu 1, -[[AMT:[0-9]+]](1)
; PPC64: addi 1, 1, [[AMT]]
; PPC64: lwz 12, 8(1)
; PPC64: mtocrf 32, 12
; PPC64: mtocrf 16, 12
; PPC64: mtocrf 8, 12

; Generate mfocrf in prologue when we need to save 1 nonvolatile CR field
define void @cloberOneNvCrField() {
entry:
  tail call void asm sideeffect "# clobbers", "~{cr2}"()
  ret void

; PPC64-ELFv2-LABEL: @cloberOneNvCrField
; PPC64-ELFv2: mfocrf [[REG1:[0-9]+]], 32
}

; Generate mfcr in prologue when we need to save all nonvolatile CR field
define void @cloberAllNvCrField() {
entry:
  tail call void asm sideeffect "# clobbers", "~{cr2},~{cr3},~{cr4}"()
  ret void

; PPC64-ELFv2-LABEL: @cloberAllNvCrField
; PPC64-ELFv2: mfcr [[REG1:[0-9]+]]
}

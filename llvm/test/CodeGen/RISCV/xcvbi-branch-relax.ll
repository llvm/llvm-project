; NOTE: Do not use update_llc_test_checks.py on this file.
; NOTE: Branch relaxation for RISC-V happens at the MC layer in RISCVAsmBackend, not in a MIR pass.
; It is therefore only observable on object emission, not on textual asm. Use -filetype=obj + llvm-objdump.

; RUN: llc -mtriple=riscv32 -mattr=+xcvbi -filetype=obj -verify-machineinstrs < %s \
; RUN:   | llvm-objdump -d -M no-aliases --no-show-raw-insn - | FileCheck %s

; A short forward branch (target ~8 KiB away, well past the +/-4094 range
; of cv.beqimm / cv.bneimm) must be relaxed by the MC layer into:
;     <inverted short branch>  rs1, imm5, .Lskip
;     jal                      zero, target
;   .Lskip:
; Without relaxation, the assembler would silently truncate the 13-bit
; branch immediate and produce wrong code.

define void @test_far_beqimm(i32 %x) {
; CHECK-LABEL: <test_far_beqimm>:
; CHECK:       cv.bneimm a0, 0x3, 0x{{[0-9a-f]+}}
; CHECK-NEXT:  jal zero, 0x{{[0-9a-f]+}}
entry:
  %cmp = icmp eq i32 %x, 3
  br i1 %cmp, label %target, label %filler

filler:
  ; ~8 KiB of dead bytes via inline asm, to force the branch out of range.
  call void asm sideeffect ".zero 8192", ""()
  br label %target

target:
  ret void
}

define void @test_far_bneimm(i32 %x) {
; CHECK-LABEL: <test_far_bneimm>:
; CHECK:       cv.beqimm a0, 0x3, 0x{{[0-9a-f]+}}
; CHECK-NEXT:  jal zero, 0x{{[0-9a-f]+}}
entry:
  %cmp = icmp ne i32 %x, 3
  br i1 %cmp, label %target, label %filler

filler:
  ; ~8 KiB of dead bytes via inline asm, to force the branch out of range.
  call void asm sideeffect ".zero 8192", ""()
  br label %target

target:
  ret void
}

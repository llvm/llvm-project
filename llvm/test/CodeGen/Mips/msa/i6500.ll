; Test the MSA intrinsics that are encoded with the SPECIAL instruction format.

; RUN: llc -mtriple=mips-elf -mcpu=i6500 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips64-elf -mcpu=i6500 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS64
; RUN: llc -mtriple=mips-elf -mcpu=i6500 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips64-elf -mcpu=i6500 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS64
; RUN: llc -mtriple=mips64-elf -mcpu=i6500 -mattr=-msa < %s | \
; RUN:   FileCheck %s --check-prefix=NO-DSLA
; RUN: llc -mtriple=mips-elf -mcpu=i6400 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips64-elf -mcpu=i6400 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS64
; RUN: llc -mtriple=mips-elf -mcpu=i6400 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips64-elf -mcpu=i6400 < %s | \
; RUN:   FileCheck %s --check-prefix=MIPS64
; RUN: llc -mtriple=mips64-elf -mcpu=i6400 -mattr=-msa < %s | \
; RUN:   FileCheck %s --check-prefix=NO-DSLA

define i32 @llvm_mips_lsa_test(i32 %a, i32 %b) nounwind {
entry:
  %0 = tail call i32 @llvm.mips.lsa(i32 %a, i32 %b, i32 2)
  ret i32 %0
}

declare i32 @llvm.mips.lsa(i32, i32, i32) nounwind

; MIPS32: llvm_mips_lsa_test:
; MIPS32: lsa {{\$[0-9]+}}, $5, $4, 2
; MIPS32: .size llvm_mips_lsa_test

define i32 @lsa_test(i32 %a, i32 %b) nounwind {
entry:
  %0 = shl i32 %b, 2
  %1 = add i32 %a, %0
  ret i32 %1
}

; MIPS32: lsa_test:
; MIPS32: lsa {{\$[0-9]+}}, $5, $4, 2
; MIPS32: .size lsa_test

define i64 @llvm_mips_dlsa_test(i64 %a, i64 %b) nounwind {
entry:
  %0 = tail call i64 @llvm.mips.dlsa(i64 %a, i64 %b, i32 2)
  ret i64 %0
}

declare i64 @llvm.mips.dlsa(i64, i64, i32) nounwind

; MIPS64: llvm_mips_dlsa_test:
; MIPS64: dlsa {{\$[0-9]+}}, $5, $4, 2
; MIPS64: .size llvm_mips_dlsa_test
; NO-DSLA-NOT: dlsa {{\$[0-9]+}}, $5, $4, 2
define i64 @dlsa_test(i64 %a, i64 %b) nounwind {
entry:
  %0 = shl i64 %b, 2
  %1 = add i64 %a, %0
  ret i64 %1
}

; MIPS64: dlsa_test:
; MIPS64: dlsa {{\$[0-9]+}}, $5, $4, 2
; MIPS64: .size dlsa_test
; NO-DSLA-NOT: dlsa {{\$[0-9]+}}, $5, $4, 2

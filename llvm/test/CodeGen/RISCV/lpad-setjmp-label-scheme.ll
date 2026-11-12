; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp \
; RUN:   -verify-machineinstrs < %t/unlabeled.ll | FileCheck %s --check-prefixes=NOOVERRIDE-UNLABELED
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp \
; RUN:   -verify-machineinstrs < %t/funcsig.ll | FileCheck %s --check-prefixes=NOOVERRIDE-FUNCSIG
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp \
; RUN:   -riscv-landing-pad-label=778899 \
; RUN:   -verify-machineinstrs < %t/unlabeled.ll | FileCheck %s --check-prefixes=OVERRIDE-UNLABELED
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp \
; RUN:   -riscv-landing-pad-label=778899 \
; RUN:   -verify-machineinstrs < %t/funcsig.ll | FileCheck %s --check-prefixes=OVERRIDE-FUNCSIG

;; PseudoCALLLpadAlign/PseudoCALLIndirectLpadAlign do not have call / entry 
;; labels set up by RISCVLandingPadSetup/RISCVIndirectBranchTracking.
;; this is because it is unclear how the func-sig scheme should generate
;; and handle these labels.
;; this test essentially pins legacy behaviour (also in regards to
;; -riscv-landing-pad-label) until it is determined how the func-sig scheme 
;; should deal with PseudoCALLLpadAlign/PseudoCALLIndirectLpadAlign
;; FIXME: if func-sig starts handling these, this test needs updating.

;--- unlabeled.ll
declare i32 @setjmp(ptr) returns_twice
declare ptr @malloc(i64)

define i32 @test_direct_returns_twice() {
; NOOVERRIDE-UNLABELED-LABEL: test_direct_returns_twice:
; NOOVERRIDE-UNLABELED:       # %bb.0:
; NOOVERRIDE-UNLABELED-NEXT:    lpad 0
; NOOVERRIDE-UNLABELED-NEXT:    addi sp, sp, -16
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 16
; NOOVERRIDE-UNLABELED-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_offset ra, -8
; NOOVERRIDE-UNLABELED-NEXT:    li a0, 8
; NOOVERRIDE-UNLABELED-NEXT:    call malloc
; NOOVERRIDE-UNLABELED-NEXT:    call setjmp
; NOOVERRIDE-UNLABELED-NEXT:    lpad 0
; NOOVERRIDE-UNLABELED-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_restore ra
; NOOVERRIDE-UNLABELED-NEXT:    addi sp, sp, 16
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 0
; NOOVERRIDE-UNLABELED-NEXT:    ret
;
; OVERRIDE-UNLABELED-LABEL: test_direct_returns_twice:
; OVERRIDE-UNLABELED:       # %bb.0:
; OVERRIDE-UNLABELED-NEXT:    lpad 778899
; OVERRIDE-UNLABELED-NEXT:    addi sp, sp, -16
; OVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 16
; OVERRIDE-UNLABELED-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; OVERRIDE-UNLABELED-NEXT:    .cfi_offset ra, -8
; OVERRIDE-UNLABELED-NEXT:    li a0, 8
; OVERRIDE-UNLABELED-NEXT:    call malloc
; OVERRIDE-UNLABELED-NEXT:    call setjmp
; OVERRIDE-UNLABELED-NEXT:    lpad 778899
; OVERRIDE-UNLABELED-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; OVERRIDE-UNLABELED-NEXT:    .cfi_restore ra
; OVERRIDE-UNLABELED-NEXT:    addi sp, sp, 16
; OVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 0
; OVERRIDE-UNLABELED-NEXT:    ret
  %1 = call ptr @malloc(i64 8)
  %2 = call i32 @setjmp(ptr %1) returns_twice
  ret i32 %2
}

define i32 @test_indirect_returns_twice(ptr %fptr, ptr %buf) {
; NOOVERRIDE-UNLABELED-LABEL: test_indirect_returns_twice:
; NOOVERRIDE-UNLABELED:       # %bb.0:
; NOOVERRIDE-UNLABELED-NEXT:    lpad 0
; NOOVERRIDE-UNLABELED-NEXT:    addi sp, sp, -16
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 16
; NOOVERRIDE-UNLABELED-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_offset ra, -8
; NOOVERRIDE-UNLABELED-NEXT:    mv a2, a0
; NOOVERRIDE-UNLABELED-NEXT:    mv a0, a1
; NOOVERRIDE-UNLABELED-NEXT:    jalr a2
; NOOVERRIDE-UNLABELED-NEXT:    lpad 0
; NOOVERRIDE-UNLABELED-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_restore ra
; NOOVERRIDE-UNLABELED-NEXT:    addi sp, sp, 16
; NOOVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 0
; NOOVERRIDE-UNLABELED-NEXT:    ret
;
; OVERRIDE-UNLABELED-LABEL: test_indirect_returns_twice:
; OVERRIDE-UNLABELED:       # %bb.0:
; OVERRIDE-UNLABELED-NEXT:    lpad 778899
; OVERRIDE-UNLABELED-NEXT:    addi sp, sp, -16
; OVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 16
; OVERRIDE-UNLABELED-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; OVERRIDE-UNLABELED-NEXT:    .cfi_offset ra, -8
; OVERRIDE-UNLABELED-NEXT:    mv a2, a0
; OVERRIDE-UNLABELED-NEXT:    mv a0, a1
; OVERRIDE-UNLABELED-NEXT:    jalr a2
; OVERRIDE-UNLABELED-NEXT:    lpad 778899
; OVERRIDE-UNLABELED-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; OVERRIDE-UNLABELED-NEXT:    .cfi_restore ra
; OVERRIDE-UNLABELED-NEXT:    addi sp, sp, 16
; OVERRIDE-UNLABELED-NEXT:    .cfi_def_cfa_offset 0
; OVERRIDE-UNLABELED-NEXT:    ret
  %1 = call i32 %fptr(ptr %buf) returns_twice
  ret i32 %1
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}

;--- funcsig.ll
declare i32 @setjmp(ptr) returns_twice
declare ptr @malloc(i64)

define i32 @test_direct_returns_twice() {
; NOOVERRIDE-FUNCSIG-LABEL: test_direct_returns_twice:
; NOOVERRIDE-FUNCSIG:       # %bb.0:
; NOOVERRIDE-FUNCSIG-NEXT:    lpad 0
; NOOVERRIDE-FUNCSIG-NEXT:    addi sp, sp, -16
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 16
; NOOVERRIDE-FUNCSIG-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_offset ra, -8
; NOOVERRIDE-FUNCSIG-NEXT:    li a0, 8
; NOOVERRIDE-FUNCSIG-NEXT:    call malloc
; NOOVERRIDE-FUNCSIG-NEXT:    call setjmp
; NOOVERRIDE-FUNCSIG-NEXT:    lpad 0
; NOOVERRIDE-FUNCSIG-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_restore ra
; NOOVERRIDE-FUNCSIG-NEXT:    addi sp, sp, 16
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 0
; NOOVERRIDE-FUNCSIG-NEXT:    ret
;
; OVERRIDE-FUNCSIG-LABEL: test_direct_returns_twice:
; OVERRIDE-FUNCSIG:       # %bb.0:
; OVERRIDE-FUNCSIG-NEXT:    lpad 0
; OVERRIDE-FUNCSIG-NEXT:    addi sp, sp, -16
; OVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 16
; OVERRIDE-FUNCSIG-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; OVERRIDE-FUNCSIG-NEXT:    .cfi_offset ra, -8
; OVERRIDE-FUNCSIG-NEXT:    li a0, 8
; OVERRIDE-FUNCSIG-NEXT:    call malloc
; OVERRIDE-FUNCSIG-NEXT:    call setjmp
; OVERRIDE-FUNCSIG-NEXT:    lpad 778899
; OVERRIDE-FUNCSIG-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; OVERRIDE-FUNCSIG-NEXT:    .cfi_restore ra
; OVERRIDE-FUNCSIG-NEXT:    addi sp, sp, 16
; OVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 0
; OVERRIDE-FUNCSIG-NEXT:    ret
  %1 = call ptr @malloc(i64 8)
  %2 = call i32 @setjmp(ptr %1) returns_twice
  ret i32 %2
}

define i32 @test_indirect_returns_twice(ptr %fptr, ptr %buf) {
; NOOVERRIDE-FUNCSIG-LABEL: test_indirect_returns_twice:
; NOOVERRIDE-FUNCSIG:       # %bb.0:
; NOOVERRIDE-FUNCSIG-NEXT:    lpad 0
; NOOVERRIDE-FUNCSIG-NEXT:    addi sp, sp, -16
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 16
; NOOVERRIDE-FUNCSIG-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_offset ra, -8
; NOOVERRIDE-FUNCSIG-NEXT:    mv a2, a0
; NOOVERRIDE-FUNCSIG-NEXT:    mv a0, a1
; NOOVERRIDE-FUNCSIG-NEXT:    jalr a2
; NOOVERRIDE-FUNCSIG-NEXT:    lpad 0
; NOOVERRIDE-FUNCSIG-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_restore ra
; NOOVERRIDE-FUNCSIG-NEXT:    addi sp, sp, 16
; NOOVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 0
; NOOVERRIDE-FUNCSIG-NEXT:    ret
;
; OVERRIDE-FUNCSIG-LABEL: test_indirect_returns_twice:
; OVERRIDE-FUNCSIG:       # %bb.0:
; OVERRIDE-FUNCSIG-NEXT:    lpad 0
; OVERRIDE-FUNCSIG-NEXT:    addi sp, sp, -16
; OVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 16
; OVERRIDE-FUNCSIG-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; OVERRIDE-FUNCSIG-NEXT:    .cfi_offset ra, -8
; OVERRIDE-FUNCSIG-NEXT:    mv a2, a0
; OVERRIDE-FUNCSIG-NEXT:    mv a0, a1
; OVERRIDE-FUNCSIG-NEXT:    jalr a2
; OVERRIDE-FUNCSIG-NEXT:    lpad 778899
; OVERRIDE-FUNCSIG-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; OVERRIDE-FUNCSIG-NEXT:    .cfi_restore ra
; OVERRIDE-FUNCSIG-NEXT:    addi sp, sp, 16
; OVERRIDE-FUNCSIG-NEXT:    .cfi_def_cfa_offset 0
; OVERRIDE-FUNCSIG-NEXT:    ret
  %1 = call i32 %fptr(ptr %buf) returns_twice
  ret i32 %1
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
; RUN: llc -mtriple=riscv32 < %s | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple=riscv32 -mattr=+zbb < %s | FileCheck %s --check-prefix=RV32ZBB
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s --check-prefix=RV64
; RUN: llc -mtriple=riscv64 -mattr=+zbb < %s | FileCheck %s --check-prefix=RV64ZBB

define i16 @test_bitextract_var(b32 %src, i32 %off) {
; RV32-LABEL: test_bitextract_var:
; RV32:       srl  a0, a0, a1
; RV32:       ret

; RV32ZBB-LABEL: test_bitextract_var:
; RV32ZBB:       srl  a0, a0, a1
; RV32ZBB:       ret

; RV64-LABEL: test_bitextract_var:
; RV64:       srlw  a0, a0, a1
; RV64:       ret

; RV64ZBB-LABEL: test_bitextract_var:
; RV64ZBB:       srlw  a0, a0, a1
; RV64ZBB:       ret
  %result = bitextract i16, b32 %src, i32 %off
  ret i16 %result
}

define i16 @test_bitextract_const(b32 %src) {
; RV32-LABEL: test_bitextract_const:
; RV32:       srli a0, a0, 8
; RV32:       ret

; RV32ZBB-LABEL: test_bitextract_const:
; RV32ZBB:       srli a0, a0, 8
; RV32ZBB:       ret

; RV64-LABEL: test_bitextract_const:
; RV64:       srliw a0, a0, 8
; RV64:       ret

; RV64ZBB-LABEL: test_bitextract_const:
; RV64ZBB:       srliw a0, a0, 8
; RV64ZBB:       ret
  %result = bitextract i16, b32 %src, i32 8
  ret i16 %result
}

define i8 @test_bitextract_narrow(b64 %src) {
; RV32-LABEL: test_bitextract_narrow:
; RV32:       ret

; RV32ZBB-LABEL: test_bitextract_narrow:
; RV32ZBB:       ret

; RV64-LABEL: test_bitextract_narrow:
; RV64:       ret

; RV64ZBB-LABEL: test_bitextract_narrow:
; RV64ZBB:       ret
  %result = bitextract i8, b64 %src, i32 0
  ret i8 %result
}

define b32 @test_bitinsert_var(b32 %base, i16 %val, i32 %off) {
; RV32-LABEL: test_bitinsert_var:
; RV32	        srl	a3, a0, a2
; RV32	        neg	a4, a2
; RV32	        sll	a0, a0, a4
; RV32	        or	a0, a3, a0
; RV32	        lui	a3, 1048560
; RV32	        slli	a1, a1, 16
; RV32	        srli	a1, a1, 16
; RV32	        and	a0, a0, a3
; RV32	        or	a0, a0, a1
; RV32	        srl	a1, a0, a4
; RV32	        sll	a0, a0, a2
; RV32	        or	a0, a0, a1
; RV32	        ret


; RV32ZBB-LABEL: test_bitinsert_var:
; RV32ZBB:       ror	a0, a0, a2
; RV32ZBB:       lui	a3, 1048560
; RV32ZBB:       and	a0, a0, a3
; RV32ZBB:       zext.h	a1, a1
; RV32ZBB:       or	a0, a0, a1
; RV32ZBB:       rol	a0, a0, a2
; RV32ZBB:       ret


; RV64-LABEL: test_bitinsert_var:
; RV64:	      srlw	a3, a0, a2
; RV64:	      neg	a4, a2
; RV64:	      sllw	a0, a0, a4
; RV64:	      or	a0, a3, a0
; RV64:	      lui	a3, 1048560
; RV64:	      slli	a1, a1, 48
; RV64:	      srli	a1, a1, 48
; RV64:	      and	a0, a0, a3
; RV64:	      or	a0, a0, a1
; RV64:	      srlw	a1, a0, a4
; RV64:	      sllw	a0, a0, a2
; RV64:	      or	a0, a0, a1
; RV64:	      ret


; RV64ZBB-LABEL: test_bitinsert_var:
; RV64ZBB:       rorw	a0, a0, a2
; RV64ZBB:       lui	a3, 1048560
; RV64ZBB:       and	a0, a0, a3
; RV64ZBB:       zext.h	a1, a1
; RV64ZBB:       or	a0, a0, a1
; RV64ZBB:       rolw	a0, a0, a2
; RV64ZBB:       ret

  %result = bitinsert b32 %base, i16 %val, i32 %off
  ret b32 %result
}

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=M16
; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32  -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=M32

;; When in MIPS16 mode, LLVM used to generate the global base register 
;; initialization code for both MIPS32 and MIPS16. The MIPS32 version
;; uses the LUI instruction that is not supported in MIPS16. This checks
;; that only the expected sequences are generated for MIPS32 vs MIPS16.

@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret i32 0

; M16:      # %bb.0:
; M16-NOT:       lui     ${{[0-9]+}}, %hi(_gp_disp)
; M16-NOT:       addiu   ${{[0-9]+}}, ${{[0-9]+}}, %lo(_gp_disp)

; M16:           li      $[[R1:[0-9]+]], %hi(_gp_disp)
; M16:           addiu   $[[R2:[0-9]+]], $pc, %lo(_gp_disp)
; M16:           sll     $[[R3:[0-9]+]], $[[R1]], 16
; M16:           addu    ${{[0-9]+}}, $[[R2]], $[[R3]]


; M32:      # %bb.0:
; M32:           lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; M32:           addiu   $[[R0]], $[[R0]], %lo(_gp_disp)

; M32-NOT:       li      ${{[0-9]+}}, %hi(_gp_disp)
; M32-NOT:       addiu   ${{[0-9]+}}, $pc, %lo(_gp_disp)
; M32-NOT:       sll     ${{[0-9]+}}, ${{[0-9]+}}, 16
; M32-NOT:       addu    ${{[0-9]+}}, ${{[0-9]+}}, ${{[0-9]+}}

}

declare i32 @printf(ptr, ...)

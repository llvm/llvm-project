; RUN: llc -mtriple=mipsel-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=mipsel-unknown-linux-gnu -relocation-model=pic -filetype=obj -o %t %s
; RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
; RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

declare i32 @"\01my_target_sym"()

define i32 @caller() nounwind {
; ASM-LABEL: caller:
; ASM:       # %bb.0: # %entry
; ASM-NEXT:    lui $2, %hi(_gp_disp)
; ASM-NEXT:    addiu $2, $2, %lo(_gp_disp)
; ASM-NEXT:    addiu $sp, $sp, -24
; ASM-NEXT:    sw $ra, 20($sp)
; ASM-NEXT:    addu $gp, $2, $25
; ASM-NEXT:    lw $25, %call16(my_target_sym)($gp)
; ASM-NEXT:    .reloc $tmp0, R_MIPS_JALR, my_target_sym
; ASM-NEXT:  $tmp0:
; ASM-NEXT:    jalr $25
; ASM-NEXT:    nop

; RELOC: R_MIPS_JALR{{.*}}my_target_sym

; SYM: UND{{.*}}my_target_sym
; SYM-NOT: UND{{.*}}my_target_sym

entry:
  %call = call i32 @"\01my_target_sym"()
  ret i32 %call
}

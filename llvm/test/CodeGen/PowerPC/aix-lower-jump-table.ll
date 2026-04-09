; RUN: llc -mtriple powerpc-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -code-model=small < %s | FileCheck \
; RUN: --check-prefixes=32SMALL-ASM,SMALL-ASM %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -code-model=large < %s | FileCheck \
; RUN: --check-prefixes=32LARGE-ASM,LARGE-ASM %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -code-model=small < %s | FileCheck \
; RUN: --check-prefixes=64SMALL-ASM,SMALL-ASM %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -code-model=large < %s | FileCheck \
; RUN: --check-prefixes=64LARGE-ASM,LARGE-ASM %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -function-sections < %s | FileCheck \
; RUN: --check-prefix=FUNC-ASM %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -function-sections < %s | FileCheck \
; RUN: --check-prefix=FUNC-ASM %s

define i32 @jump_table(i32 %a) {
entry:
  switch i32 %a, label %sw.epilog [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
  ]

sw.bb:
  tail call void asm sideeffect "", ""()
  br label %sw.epilog

sw.bb1:
  tail call void asm sideeffect "", ""()
  br label %sw.epilog

sw.bb2:
  tail call void asm sideeffect "", ""()
  br label %sw.epilog

sw.bb3:
  tail call void asm sideeffect "", ""()
  br label %sw.epilog

sw.epilog:
  ret i32 0
}
; 32SMALL-ASM-LABEL: jump_table
; 32SMALL-ASM: .jump_table:
; 32SMALL-ASM:      addi 3, 3, -1
; 32SMALL-ASM: 	    cmplwi 3, 3
; 32SMALL-ASM: 	    bgt	0, L..BB0_3
; 32SMALL-ASM: 	    lwz 4, L..C0(2)
; 32SMALL-ASM: 	    slwi 3, 3, 2
; 32SMALL-ASM: 	    lwzx 3, 4, 3
; 32SMALL-ASM: 	    add 3, 4, 3
; 32SMALL-ASM: 	    mtctr 3
; 32SMALL-ASM: 	    bctr
; 32SMALL-ASM: L..BB0_2:
; 32SMALL-ASM: L..BB0_3:
; 32SMALL-ASM: L..BB0_4:
; 32SMALL-ASM: L..BB0_5:
; 32SMALL-ASM: L..BB0_6:
; 32SMALL-ASM: 	    li 3, 0
; 32SMALL-ASM: 	    blr
; 32SMALL-ASM: 	    .csect .rodata[RO],2
; 32SMALL-ASM: 	    .align  2
; 32SMALL-ASM: L..JTI0_0:
; 32SMALL-ASM: 	    .vbyte	4, L..BB0_2-L..JTI0_0
; 32SMALL-ASM: 	    .vbyte	4, L..BB0_6-L..JTI0_0
; 32SMALL-ASM: 	    .vbyte	4, L..BB0_4-L..JTI0_0
; 32SMALL-ASM: 	    .vbyte	4, L..BB0_5-L..JTI0_0

; 32LARGE-ASM-LABEL: jump_table
; 32LARGE-ASM: .jump_table:
; 32LARGE-ASM:      addi 3, 3, -1
; 32LARGE-ASM:      cmplwi  3, 3
; 32LARGE-ASM:      bgt     0, L..BB0_3
; 32LARGE-ASM: 	    addis 4, L..C0@u(2)
; 32LARGE-ASM: 	    slwi 3, 3, 2
; 32LARGE-ASM:      lwz 4, L..C0@l(4)
; 32LARGE-ASM:      lwzx 3, 4, 3
; 32LARGE-ASM:      add 3, 4, 3
; 32LARGE-ASM:      mtctr 3
; 32LARGE-ASM:      bctr
; 32LARGE-ASM: L..BB0_2:
; 32LARGE-ASM: L..BB0_3:
; 32LARGE-ASM: L..BB0_4:
; 32LARGE-ASM: L..BB0_5:
; 32LARGE-ASM: L..BB0_6:
; 32LARGE-ASM:      li 3, 0
; 32LARGE-ASM:      blr
; 32LARGE-ASM:      .csect .rodata[RO],2
; 32LARGE-ASM:      .align  2
; 32LARGE-ASM: L..JTI0_0:
; 32LARGE-ASM:      .vbyte	4, L..BB0_2-L..JTI0_0
; 32LARGE-ASM:      .vbyte	4, L..BB0_6-L..JTI0_0
; 32LARGE-ASM:      .vbyte	4, L..BB0_4-L..JTI0_0
; 32LARGE-ASM:      .vbyte	4, L..BB0_5-L..JTI0_0

; 64SMALL-ASM-LABEL: jump_table
; 64SMALL-ASM: .jump_table:
; 64SMALL-ASM:      addi 3, 3, -1
; 64SMALL-ASM:      cmplwi  3, 3
; 64SMALL-ASM:      bgt     0, L..BB0_3
; 64SMALL-ASM:      ld 4, L..C0(2)
; 64SMALL-ASM:      rldic 3, 3, 2, 30
; 64SMALL-ASM:      lwax 3, 4, 3
; 64SMALL-ASM:      add 3, 4, 3
; 64SMALL-ASM:      mtctr 3
; 64SMALL-ASM:      bctr
; 64SMALL-ASM: L..BB0_2:
; 64SMALL-ASM: L..BB0_3:
; 64SMALL-ASM: L..BB0_4:
; 64SMALL-ASM: L..BB0_5:
; 64SMALL-ASM: L..BB0_6:
; 64SMALL-ASM:      li 3, 0
; 64SMALL-ASM:      blr
; 64SMALL-ASM:      .csect .rodata[RO],2
; 64SMALL-ASM:      .align  2
; 64SMALL-ASM: L..JTI0_0:
; 64SMALL-ASM:      .vbyte	4, L..BB0_2-L..JTI0_0
; 64SMALL-ASM:      .vbyte	4, L..BB0_6-L..JTI0_0
; 64SMALL-ASM:      .vbyte	4, L..BB0_4-L..JTI0_0
; 64SMALL-ASM:      .vbyte	4, L..BB0_5-L..JTI0_0

; 64LARGE-ASM-LABEL: jump_table
; 64LARGE-ASM: .jump_table:
; 64LARGE-ASM:      addi 3, 3, -1
; 64LARGE-ASM:      cmplwi  3, 3
; 64LARGE-ASM:      bgt     0, L..BB0_3
; 64LARGE-ASM:      addis 4, L..C0@u(2)
; 64LARGE-ASM:      rldic 3, 3, 2, 30
; 64LARGE-ASM:      ld 4, L..C0@l(4)
; 64LARGE-ASM:      lwax 3, 4, 3
; 64LARGE-ASM:      add 3, 4, 3
; 64LARGE-ASM:      mtctr 3
; 64LARGE-ASM:      bctr
; 64LARGE-ASM: L..BB0_2:
; 64LARGE-ASM: L..BB0_3:
; 64LARGE-ASM: L..BB0_4:
; 64LARGE-ASM: L..BB0_5:
; 64LARGE-ASM: L..BB0_6:
; 64LARGE-ASM:      li 3, 0
; 64LARGE-ASM:      blr
; 64LARGE-ASM:      .csect .rodata[RO],2
; 64LARGE-ASM:      .align  2
; 64LARGE-ASM: L..JTI0_0:
; 64LARGE-ASM:      .vbyte	4, L..BB0_2-L..JTI0_0
; 64LARGE-ASM:      .vbyte	4, L..BB0_6-L..JTI0_0
; 64LARGE-ASM:      .vbyte	4, L..BB0_4-L..JTI0_0
; 64LARGE-ASM:      .vbyte	4, L..BB0_5-L..JTI0_0

; FUNC-ASM:         .csect .jump_table[PR],5
; FUNC-ASM: L..BB0_2:
; FUNC-ASM: L..BB0_3:
; FUNC-ASM: L..BB0_4:
; FUNC-ASM: L..BB0_5:
; FUNC-ASM: L..BB0_6:
; FUNC-ASM:         li 3, 0
; FUNC-ASM:         blr
; FUNC-ASM:         .csect .rodata.jmp..jump_table[RO],2
; FUNC-ASM:         .align  2
; FUNC-ASM: L..JTI0_0:
; FUNC-ASM:         .vbyte  4, L..BB0_2-L..JTI0_0
; FUNC-ASM:         .vbyte  4, L..BB0_6-L..JTI0_0
; FUNC-ASM:         .vbyte  4, L..BB0_4-L..JTI0_0
; FUNC-ASM:         .vbyte  4, L..BB0_5-L..JTI0_0

; SMALL-ASM: .toc
; SMALL-ASM: .tc L..JTI0_0[TC],L..JTI0_0

; LARGE-ASM: .toc
; LARGE-ASM: .tc L..JTI0_0[TE],L..JTI0_0

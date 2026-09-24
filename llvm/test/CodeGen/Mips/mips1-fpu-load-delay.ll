; RUN: llc < %s -mtriple=mipsel -mcpu=mips1 -relocation-model=static | FileCheck %s -check-prefixes=ALL,MIPS1
; RUN: llc < %s -mtriple=mipsel -mcpu=mips2 -relocation-model=static | FileCheck %s -check-prefixes=ALL,MIPS2

@v = external global [4 x float]

define float @sum3() nounwind {
; ALL-LABEL: sum3:
entry:
  %0 = load float, ptr @v
  %1 = load float, ptr getelementptr inbounds ([4 x float], ptr @v, i32 0, i32 1)
  %add = fadd float %0, %1
  %2 = load float, ptr getelementptr inbounds ([4 x float], ptr @v, i32 0, i32 2)
  %add2 = fadd float %add, %2
  ret float %add2

; ALL:        lwc1 $f{{[0-9]+}}
; MIPS1:      nop
; MIPS2-NOT:  nop
; ALL:        add.s
}

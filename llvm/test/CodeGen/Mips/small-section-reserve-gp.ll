; RUN: llc -mtriple=mipsel-sde-elf -march=mipsel -relocation-model=static -mattr=+noabicalls -mgpopt < %s \
; RUN: | FileCheck %s

@i = internal unnamed_addr global i32 0, align 4

define i32 @geti() nounwind readonly {
entry:
; CHECK: lw ${{[0-9]+}}, %gp_rel(i)($gp)
  %0 = load i32, ptr @i, align 4
  ret i32 %0
}


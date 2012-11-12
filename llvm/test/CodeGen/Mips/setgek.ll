; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@k = global i32 10, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4
@r3 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32* @k, align 4
  %cmp = icmp sgt i32 %0, -32769
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4
; 16:	slti	${{[0-9]+}}, -32768
; 16:	move	${{[0-9]+}}, $t8
; 16:	xor	${{[0-9]+}}, ${{[0-9]+}}
  ret void
}

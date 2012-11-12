; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 1, align 4
@j = global i32 10, align 4
@k = global i32 1, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32* @i, align 4
  %1 = load i32* @k, align 4
  %cmp = icmp ne i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4
; 16:	xor	$[[REGISTER:[0-9]+]], ${{[0-9]+}}
; 16:	sltu	${{[0-9]+}}, $[[REGISTER]]
; 16:	move	${{[0-9]+}}, $t8
  ret void
}

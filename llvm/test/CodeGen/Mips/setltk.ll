; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@j = global i32 -5, align 4
@k = global i32 10, align 4
@l = global i32 20, align 4
@m = global i32 10, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4
@r3 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @j, align 4
  %cmp = icmp slt i32 %0, 10
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4
; 16:	slti	$[[REGISTER:[0-9]+]], 10
; 16:	move	$[[REGISTER]], $24
  ret void
}

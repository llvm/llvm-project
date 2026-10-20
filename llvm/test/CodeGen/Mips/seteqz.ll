; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16
; RUN: llc  -mtriple=mips -mcpu=mips32r6 -mattr=micromips -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=MMR6

@i = global i32 0, align 4
@j = global i32 99, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, ptr @i, align 4
  %cmp = icmp eq i32 %0, 0
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @r1, align 4
; 16:   sltiu   ${{[0-9]+}}, 1
; MMR6: sltiu   ${{[0-9]+}}, ${{[0-9]+}}, 1
; 16:   move    ${{[0-9]+}}, $24
  %1 = load i32, ptr @j, align 4
  %cmp1 = icmp eq i32 %1, 99
  %conv2 = zext i1 %cmp1 to i32
  store i32 %conv2, ptr @r2, align 4
; 16:   xor     $[[REGISTER:[0-9A-Ba-b_]+]], ${{[0-9]+}}
; 16:   sltiu   $[[REGISTER:[0-9A-Ba-b_]+]], 1
; MMR6: sltiu   ${{[0-9]+}}, ${{[0-9]+}}, 1
; 16:   move    ${{[0-9]+}}, $24
  ret void
}

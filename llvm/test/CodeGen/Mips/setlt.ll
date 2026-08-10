; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16
; RUN: llc  -mtriple=mips -mcpu=mips32r6 -mattr=micromips -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=MMR6

@j = global i32 -5, align 4
@k = global i32 10, align 4
@l = global i32 20, align 4
@m = global i32 10, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4
@r3 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, ptr @j, align 4
  %1 = load i32, ptr @k, align 4
  %cmp = icmp slt i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @r1, align 4
; 16:   slt     ${{[0-9]+}}, ${{[0-9]+}}
; MMR6: slt     ${{[0-9]+}}, ${{[0-9]+}}, ${{[0-9]+}}
; 16:   move    ${{[0-9]+}}, $24
  ret void
}

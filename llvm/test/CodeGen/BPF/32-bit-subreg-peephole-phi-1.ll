; RUN: llc -O2 -mtriple=bpfel -mcpu=v2 -mattr=+alu32 < %s | FileCheck %s
;
; For the below test case, 'b' in 'ret == b' needs SLL/SLR.
; 'ret' in 'ret == b' does not need SLL/SLR as all 'ret' values
; are assigned through 'w<reg> = <value>' alu32 operations.
;
; extern int helper(int);
; int test(int a, int b, int c, int d) {
;   int ret;
;   if (a < b)
;     ret = (c < d) ? -1 : 0;
;   else
;     ret = (c < a) ? 1 : 2;
;   return helper(ret == b);
; }

define dso_local i32 @test(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr {
entry:
  %cmp = icmp slt i32 %a, %b
  %cmp1 = icmp slt i32 %c, %d
  %cond = sext i1 %cmp1 to i32
  %cmp2 = icmp slt i32 %c, %a
  %cond3 = select i1 %cmp2, i32 1, i32 2
  %ret.0 = select i1 %cmp, i32 %cond, i32 %cond3
  %cmp4 = icmp eq i32 %ret.0, %b
  %conv = zext i1 %cmp4 to i32
  %call = tail call i32 @helper(i32 %conv)
  ret i32 %call
}
; CHECK: r{{[0-9]+}} = w{{[0-9]+}}
; CHECK-NOT: r{{[0-9]+}} >>= 32
; CHECK: if r{{[0-9]+}} == r{{[0-9]+}} goto

declare dso_local i32 @helper(i32) local_unnamed_addr

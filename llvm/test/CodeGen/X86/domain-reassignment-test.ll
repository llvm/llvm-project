; RUN: llc -mcpu=skylake-avx512 -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s
; RUN: llc -mcpu=skylake-avx512 -mtriple=x86_64-unknown-linux-gnu %s -o - | llvm-mc -triple=x86_64-unknown-linux-gnu

; Check that the X86 domain reassignment pass doesn't introduce an illegal
; test instruction. See PR37396
define void @japi1_foo2_34617() {
pass2:
  br label %if5

L174:
  %tmp = icmp sgt <2 x i64> undef, zeroinitializer
  %tmp1 = icmp sle <2 x i64> undef, undef
  %tmp2 = and <2 x i1> %tmp, %tmp1
  %tmp3 = extractelement <2 x i1> %tmp2, i32 0
  %tmp4 = extractelement <2 x i1> %tmp2, i32 1
  %tmp106 = and i1 %tmp4, %tmp3
  %tmp107 = zext i1 %tmp106 to i8
  %tmp108 = and i8 %tmp122, %tmp107
  %tmp109 = icmp eq i8 %tmp108, 0
; CHECK-NOT: testb  {{%k[0-7]}}
  br i1 %tmp109, label %L188, label %L190

if5:
  %b.055 = phi i8 [ 1, %pass2 ], [ %tmp122, %if5 ]
  %tmp118 = icmp sgt i64 undef, 0
  %tmp119 = icmp sle i64 undef, undef
  %tmp120 = and i1 %tmp118, %tmp119
  %tmp121 = zext i1 %tmp120 to i8
  %tmp122 = and i8 %b.055, %tmp121
  br i1 undef, label %L174, label %if5

L188:
  unreachable

L190:
  ret void
}

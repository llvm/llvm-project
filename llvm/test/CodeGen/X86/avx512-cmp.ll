; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vucomisd {{.*}}encoding: [0x62
define double @test1(double %a, double %b) nounwind {
  %tobool = fcmp une double %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub double %a, %b
  ret double %c
l2:
  %c1 = fadd double %a, %b
  ret double %c1
}

; CHECK-LABEL: test2
; CHECK: vucomiss {{.*}}encoding: [0x62
define float @test2(float %a, float %b) nounwind {
  %tobool = fcmp olt float %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub float %a, %b
  ret float %c
l2:
  %c1 = fadd float %a, %b
  ret float %c1
}

; FIXME: Can use vcmpeqss and extract from the mask here in AVX512.
; CHECK-LABEL: test3
; CHECK: vucomiss {{.*}}encoding: [0x62
define i32 @test3(float %a, float %b) {

  %cmp10.i = fcmp oeq float %a, %b
  %conv11.i = zext i1 %cmp10.i to i32
  ret i32 %conv11.i
}

; CHECK-LABEL: test5
; CHECK: ret
define float @test5(float %p) #0 {
entry:
  %cmp = fcmp oeq float %p, 0.000000e+00
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp1 = fcmp ogt float %p, 0.000000e+00
  %cond = select i1 %cmp1, float 1.000000e+00, float -1.000000e+00
  br label %return

return:                                           ; preds = %if.end, %entry
  %retval.0 = phi float [ %cond, %if.end ], [ %p, %entry ]
  ret float %retval.0
}

; CHECK-LABEL: test6
; CHECK: cmpl
; CHECK-NOT: kmov
; CHECK: ret
define i32 @test6(i32 %a, i32 %b) {
  %cmp = icmp eq i32 %a, %b
  %res = zext i1 %cmp to i32
  ret i32 %res
}

; CHECK-LABEL: test7
; CHECK: vucomisd
; CHECK-NOT: kmov
; CHECK: ret
define i32 @test7(double %x, double %y) #2 {
entry:
  %0 = fcmp one double %x, %y
  %or = zext i1 %0 to i32
  ret i32 %or
}

define i32 @test8(i32 %a1, i32 %a2, i32 %a3) {
  %tmp1 = icmp eq i32 %a1, -1
  %tmp2 = icmp eq i32 %a2, -2147483648
  %tmp3 = and i1 %tmp1, %tmp2
  %tmp4 = icmp eq i32 %a3, 0
  %tmp5 = or i1 %tmp3, %tmp4
  %res = select i1 %tmp5, i32 1, i32 %a3
  ret i32 %res
}

; CHECK-LABEL: test9
; CHECK: testb
; CHECK-NOT: kmov
; CHECK: ret
define i32 @test9(i64 %a) {
 %b = and i64 %a, 1
 %cmp10.i = icmp eq i64 %b, 0
 br i1 %cmp10.i, label %A, label %B
A:
 ret i32 6
B:
 ret i32 7
}

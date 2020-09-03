; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; CHECK-LABEL: @test_nop
define void @test_nop(i32 %n) {
; CHECK: udiv i32 %n, 100
  %div = udiv i32 %n, 100
  ret void
}

; CHECK-LABEL: @test1(
define void @test1(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 65535
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: urem i16
  %div = urem i32 %n, 100
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 65536
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: urem i32 %n, 100
  %div = urem i32 %n, 100
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test3(
define void @test3(i32 %m, i32 %n) {
entry:
  %cmp1 = icmp ult i32 %m, 65535
  %cmp2 = icmp ult i32 %n, 65535
  %cmp = and i1 %cmp1, %cmp2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: urem i16
  %div = urem i32 %m, %n
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i32 %m, i32 %n) {
entry:
  %cmp1 = icmp ult i32 %m, 65535
  %cmp2 = icmp ule i32 %n, 65536
  %cmp = and i1 %cmp1, %cmp2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: urem i32 %m, %n
  %div = urem i32 %m, %n
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test5
define void @test5(i32 %n) {
  %trunc = and i32 %n, 63
  ; CHECK: urem i8
  %div = urem i32 %trunc, 42
  ret void
}

; CHECK-LABEL: @test6
define void @test6(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 255
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: urem i8
  %div = srem i32 %n, 100
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @non_power_of_2
define void @non_power_of_2(i24 %n) {
  %div = urem i24 %n, 42
  ret void
}

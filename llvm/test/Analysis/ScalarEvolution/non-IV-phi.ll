; RUN: opt "-passes=print<scalar-evolution>" -disable-output < %s 2>&1 | FileCheck %s

define void @test1(i8 %t, i32 %len) {
; CHECK-LABEL: test1
; CHECK: %sphi = phi i32 [ %ext, %entry ], [ %idx.inc.ext, %loop ]
; CHECK-NEXT:  -->  (zext i8 {%t,+,1}<%loop> to i32)

 entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %sphi = phi i32 [ %ext, %entry ], [%idx.inc.ext, %loop]

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %idx.ext = zext i8 %idx to i32

  %c = icmp ult i32 %idx.inc.ext, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

define void @test2(i8 %t, i32 %len) {
; CHECK-LABEL: test2
; CHECK: %sphi = phi i32 [ %ext.mul, %entry ], [ %mul, %loop ]
; CHECK-NEXT:  -->  (4 * (zext i8 {%t,+,1}<%loop> to i32))

 entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ext.mul = mul i32 %ext, 4

  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %sphi = phi i32 [ %ext.mul, %entry ], [%mul, %loop]

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %mul = mul i32 %idx.inc.ext, 4

  %idx.ext = zext i8 %idx to i32

  %c = icmp ult i32 %idx.inc.ext, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

; RUN: opt < %s -instcombine -S | FileCheck %s

@G16 = internal constant [10 x i16] [i16 35, i16 82, i16 69, i16 81, i16 85, 
                                     i16 73, i16 82, i16 69, i16 68, i16 0]
@GD = internal constant [6 x double]
   [double -10.0, double 1.0, double 4.0, double 2.0, double -20.0, double -40.0]

define i1 @test1(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 0
  ret i1 %R
; CHECK: @test1
; CHECK-NEXT: %R = icmp eq i32 %X, 9
; CHECK-NEXT: ret i1 %R
}

define i1 @test2(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp slt i16 %Q, 85
  ret i1 %R
; CHECK: @test2
; CHECK-NEXT: %R = icmp ne i32 %X, 4
; CHECK-NEXT: ret i1 %R
}

define i1 @test3(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp oeq double %Q, 1.0
  ret i1 %R
; CHECK: @test3
; CHECK-NEXT: %R = icmp eq i32 %X, 1
; CHECK-NEXT: ret i1 %R
}

define i1 @test4(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp sle i16 %Q, 73
  ret i1 %R
; CHECK: @test4
; CHECK-NEXT: lshr i32 933, %X
; CHECK-NEXT: and i32 {{.*}}, 1
; CHECK-NEXT: %R = icmp ne i32 {{.*}}, 0
; CHECK-NEXT: ret i1 %R
}

define i1 @test5(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = icmp eq i16 %Q, 69
  ret i1 %R
; CHECK: @test5
; CHECK-NEXT: icmp eq i32 %X, 2
; CHECK-NEXT: icmp eq i32 %X, 7
; CHECK-NEXT: %R = or i1
; CHECK-NEXT: ret i1 %R
}

define i1 @test6(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp ogt double %Q, 0.0
  ret i1 %R
; CHECK: @test6
; CHECK-NEXT: add i32 %X, -1
; CHECK-NEXT: %R = icmp ult i32 {{.*}}, 3
; CHECK-NEXT: ret i1 %R
}

define i1 @test7(i32 %X) {
  %P = getelementptr inbounds [6 x double]* @GD, i32 0, i32 %X
  %Q = load double* %P
  %R = fcmp olt double %Q, 0.0
  ret i1 %R
; CHECK: @test7
; CHECK-NEXT: add i32 %X, -1
; CHECK-NEXT: %R = icmp ugt i32 {{.*}}, 2
; CHECK-NEXT: ret i1 %R
}

define i1 @test8(i32 %X) {
  %P = getelementptr inbounds [10 x i16]* @G16, i32 0, i32 %X
  %Q = load i16* %P
  %R = and i16 %Q, 3
  %S = icmp eq i16 %R, 0
  ret i1 %S
; CHECK: @test8
; CHECK-NEXT: add i32 %X, -8
; CHECK-NEXT: icmp ult i32 {{.*}}, 2
; CHECK-NEXT: ret i1
}

@GA = internal constant [4 x { i32, i32 } ] [
  { i32, i32 } { i32 1, i32 0 },
  { i32, i32 } { i32 2, i32 1 },
  { i32, i32 } { i32 3, i32 1 },
  { i32, i32 } { i32 4, i32 0 }
]

define i1 @test9(i32 %X) {
  %P = getelementptr inbounds [4 x { i32, i32 } ]* @GA, i32 0, i32 %X, i32 1
  %Q = load i32* %P
  %R = icmp eq i32 %Q, 1
  ret i1 %R
; CHECK: @test9
; CHECK-NEXT: add i32 %X, -1
; CHECK-NEXT: icmp ult i32 {{.*}}, 2
; CHECK-NEXT: ret i1
}

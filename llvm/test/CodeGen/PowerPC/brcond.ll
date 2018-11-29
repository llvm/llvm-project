; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

define signext i32 @testi32slt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32slt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32ult(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32ult
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32sle(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32sle
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32ule(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32ule
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32eq(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32eq:
; CHECK: crxor [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32sge(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32sge:
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32uge(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32uge:
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32sgt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32sgt:
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32ugt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32ugt:
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define signext i32 @testi32ne(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
; CHECK-LABEL: testi32ne:
; CHECK: creqv [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i32 %a1
iffalse:
  ret i32 %a2
}

define i64 @testi64slt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64slt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64ult(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64ult
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64sle(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64sle
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64ule(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64ule
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64eq(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64eq
; CHECK: crxor [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64sge(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64sge
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64uge(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64uge
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64sgt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64sgt
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64ugt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64ugt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define i64 @testi64ne(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
; CHECK-LABEL: testi64ne
; CHECK: creqv [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret i64 %a1
iffalse:
  ret i64 %a2
}

define float @testfloatslt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatslt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatult(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatult
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatsle(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatsle
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatule(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatule
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloateq(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloateq
; CHECK: crxor [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatsge(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatsge
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatuge(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatuge
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatsgt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatsgt
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatugt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatugt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define float @testfloatne(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
; CHECK-LABEL: testfloatne
; CHECK: creqv [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret float %a1
iffalse:
  ret float %a2
}

define double @testdoubleslt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleslt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoubleult(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleult:
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoublesle(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoublesle
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoubleule(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleule:
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoubleeq(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleeq
; CHECK: crxor [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoublesge(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoublesge
; CHECK: crandc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoubleuge(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleuge
; CHECK: crandc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoublesgt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoublesgt:
; CHECK: crorc [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoubleugt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoubleugt
; CHECK: crorc [[REG:[0-9]+]], 2, 6
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

define double @testdoublene(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
; CHECK-LABEL: testdoublene
; CHECK: creqv [[REG:[0-9]+]], 6, 2
; CHECK: bc 12, [[REG]], {{\.[a-zA-Z0-9_]+}}
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  br i1 %cmp3, label %iftrue, label %iffalse
iftrue:
  ret double %a1
iffalse:
  ret double %a2
}

; RUN: llc < %s -mcpu=pentium -mtriple=i686-linux-gnu -float-abi=soft | FileCheck %s

define i1 @test1(double %d) #0 {
entry:
  %cmp = fcmp ule double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test1:
; CHECK: calll __gtdf2
; CHECK: setle
; CHECK: retl
 
define i1 @test2(double %d) #0 {
entry:
  %cmp = fcmp ult double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test2:
; CHECK: calll __gedf2
; CHECK: sets
; CHECK: retl

define i1 @test3(double %d) #0 {
entry:
  %cmp = fcmp ugt double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test3:
; CHECK: calll __ledf2
; CHECK: setg
; CHECK: retl

define i1 @test4(double %d) #0 {
entry:
  %cmp = fcmp uge double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test4:
; CHECK: calll __ltdf2
; CHECK: setns
; CHECK: retl

define i1 @test5(double %d) #0 {
entry:
  %cmp = fcmp ole double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test5:  
; CHECK: calll __ledf2
; CHECK: setle
; CHECK: retl

define i1 @test6(double %d) #0 {
entry:
  %cmp = fcmp olt double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test6:
; CHECK: calll __ltdf2
; CHECK: sets
; CHECK: retl

define i1 @test7(double %d) #0 {
entry:
  %cmp = fcmp ogt double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test7:
; CHECK: calll __gtdf2
; CHECK: setg
; CHECK: retl

define i1 @test8(double %d) #0 {
entry:
  %cmp = fcmp oge double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test8:
; CHECK: calll __gedf2
; CHECK: setns
; CHECK: retl

define i1 @test9(double %d) #0 {
entry:
  %cmp = fcmp oeq double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test9:
; CHECK: calll __eqdf2
; CHECK: sete
; CHECK: retl

define i1 @test10(double %d) #0 {
entry:
  %cmp = fcmp ueq double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test10:
; CHECK: calll __eqdf2
; CHECK: sete
; CHECK: calll __unorddf2
; CHECK: setne
; CHECK: or
; CHECK: retl

define i1 @test11(double %d) #0 {
entry:
  %cmp = fcmp one double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test11:
; CHECK: calll __eqdf2
; CHECK: setne
; CHECK: calll __unorddf2
; CHECK: sete
; CHECK: and
; CHECK: retl

define i1 @test12(double %d) #0 {
entry:
  %cmp = fcmp une double %d, 0.000000e+00
  ret i1 %cmp
}
; CHECK-LABEL: test12:
; CHECK: calll __nedf2
; CHECK: setne
; CHECK: retl

attributes #0 = { "use-soft-float"="true" }

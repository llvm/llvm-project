; RUN: llc -march=x86 < %s | FileCheck %s

define i32 @test1(i32 %a, i32 %b) nounwind readnone {
  %not.cmp = icmp ne i32 %a, 0
  %inc = zext i1 %not.cmp to i32
  %retval.0 = add i32 %inc, %b
  ret i32 %retval.0
; CHECK: test1:
; CHECK: cmpl $1
; CHECK: sbbl $-1
; CHECK: ret
}

define i32 @test2(i32 %a, i32 %b) nounwind readnone {
  %cmp = icmp eq i32 %a, 0
  %inc = zext i1 %cmp to i32
  %retval.0 = add i32 %inc, %b
  ret i32 %retval.0
; CHECK: test2:
; CHECK: cmpl $1
; CHECK: adcl $0
; CHECK: ret
}

define i32 @test3(i32 %a, i32 %b) nounwind readnone {
  %cmp = icmp eq i32 %a, 0
  %inc = zext i1 %cmp to i32
  %retval.0 = add i32 %inc, %b
  ret i32 %retval.0
; CHECK: test3:
; CHECK: cmpl $1
; CHECK: adcl $0
; CHECK: ret
}

define i32 @test4(i32 %a, i32 %b) nounwind readnone {
  %not.cmp = icmp ne i32 %a, 0
  %inc = zext i1 %not.cmp to i32
  %retval.0 = add i32 %inc, %b
  ret i32 %retval.0
; CHECK: test4:
; CHECK: cmpl $1
; CHECK: sbbl $-1
; CHECK: ret
}

define i32 @test5(i32 %a, i32 %b) nounwind readnone {
  %not.cmp = icmp ne i32 %a, 0
  %inc = zext i1 %not.cmp to i32
  %retval.0 = sub i32 %b, %inc
  ret i32 %retval.0
; CHECK: test5:
; CHECK: cmpl $1
; CHECK: adcl $-1
; CHECK: ret
}

define i32 @test6(i32 %a, i32 %b) nounwind readnone {
  %cmp = icmp eq i32 %a, 0
  %inc = zext i1 %cmp to i32
  %retval.0 = sub i32 %b, %inc
  ret i32 %retval.0
; CHECK: test6:
; CHECK: cmpl $1
; CHECK: sbbl $0
; CHECK: ret
}

define i32 @test7(i32 %a, i32 %b) nounwind readnone {
  %cmp = icmp eq i32 %a, 0
  %inc = zext i1 %cmp to i32
  %retval.0 = sub i32 %b, %inc
  ret i32 %retval.0
; CHECK: test7:
; CHECK: cmpl $1
; CHECK: sbbl $0
; CHECK: ret
}

define i32 @test8(i32 %a, i32 %b) nounwind readnone {
  %not.cmp = icmp ne i32 %a, 0
  %inc = zext i1 %not.cmp to i32
  %retval.0 = sub i32 %b, %inc
  ret i32 %retval.0
; CHECK: test8:
; CHECK: cmpl $1
; CHECK: adcl $-1
; CHECK: ret
}

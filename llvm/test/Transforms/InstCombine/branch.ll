; Check that we fold the condition of branches of the
; form: br <condition> dest1, dest2, where dest1 == dest2.
; RUN: opt -instcombine -S < %s | FileCheck %s

define i32 @test(i32 %x) {
; CHECK-LABEL: @test
entry:
; CHECK-NOT: icmp
; CHECK: br i1 false
  %cmp = icmp ult i32 %x, 7
  br i1 %cmp, label %merge, label %merge
merge:
; CHECK-LABEL: merge:
; CHECK: ret i32 %x
  ret i32 %x
}

@global = global i8 0

define i32 @pat(i32 %x) {
; CHECK-NOT: icmp false
; CHECK: br i1 false
  %y = icmp eq i32 27, ptrtoint(i8* @global to i32)
  br i1 %y, label %patatino, label %patatino
patatino:
  ret i32 %x
}

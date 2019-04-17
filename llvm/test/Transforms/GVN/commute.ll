; RUN: opt -gvn -S < %s | FileCheck %s

declare void @use(i32, i32)

define void @foo(i32 %x, i32 %y) {
  ; CHECK-LABEL: @foo(
  %add1 = add i32 %x, %y
  %add2 = add i32 %y, %x
  call void @use(i32 %add1, i32 %add2)
  ; CHECK: @use(i32 %add1, i32 %add1)
  ret void
}

declare void @vse(i1, i1)

define void @bar(i32 %x, i32 %y) {
  ; CHECK-LABEL: @bar(
  %cmp1 = icmp ult i32 %x, %y
  %cmp2 = icmp ugt i32 %y, %x
  call void @vse(i1 %cmp1, i1 %cmp2)
  ; CHECK: @vse(i1 %cmp1, i1 %cmp1)
  ret void
}

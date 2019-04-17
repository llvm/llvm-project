; RUN: opt -reassociate -S < %s | FileCheck %s

declare void @use(i32)

define void @test1(i32 %x, i32 %y) {
; CHECK-LABEL: test1
; CHECK: mul i32 %y, %x
; CHECK: mul i32 %y, %x
; CHECK: sub i32 %1, %2
; CHECK: call void @use(i32 %{{.*}})
; CHECK: call void @use(i32 %{{.*}})

  %1 = mul i32 %x, %y
  %2 = mul i32 %y, %x
  %3 = sub i32 %1, %2
  call void @use(i32 %1)
  call void @use(i32 %3)
  ret void
}

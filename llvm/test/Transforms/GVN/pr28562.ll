; RUN: opt -S -passes=gvn < %s | FileCheck %s
define i32* @test1(i32* %a) {
  %x1 = getelementptr inbounds i32, i32* %a, i32 10
  %x2 = getelementptr i32, i32* %a, i32 10
  ret i32* %x2
; CHECK-LABEL: @test1(
; CHECK: %[[x:.*]] = getelementptr i32, i32* %a, i32 10
; CHECK: ret i32* %[[x]]
}

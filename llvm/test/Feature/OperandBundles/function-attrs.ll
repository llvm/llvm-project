; RUN: opt -S -function-attrs < %s -enable-new-pm=0 | FileCheck %s

declare void @f_readonly() readonly
declare void @f_readnone() readnone

define void @test_0(i32* %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_0(i32* %x) #2 {
 entry:
 ; CHECK: call void @f_readonly() [ "foo"(i32* %x) ]
  call void @f_readonly() [ "foo"(i32* %x) ]
  ret void
}

define void @test_1(i32* %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_1(i32* %x) #3 {
 entry:
 ; CHECK: call void @f_readnone() [ "foo"(i32* %x) ]
  call void @f_readnone() [ "foo"(i32* %x) ]
  ret void
}

define void @test_2(i32* %x) {
; The "deopt" operand bundle does not capture or write to %x.

; CHECK-LABEL: define void @test_2(i32* nocapture readonly %x)
 entry:
  call void @f_readonly() [ "deopt"(i32* %x) ]
  ret void
}

; CHECK: attributes #2 = { nofree }
; CHECK: attributes #3 = { nofree nosync }

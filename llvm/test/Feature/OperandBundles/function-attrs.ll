; RUN: opt -S -passes=inferattrs,function-attrs < %s | FileCheck %s

declare void @f_readonly() readonly
declare void @f_readnone() readnone
declare void @f_writeonly() writeonly

define void @test_0(i32* %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_0(i32* %x) #3 {
 entry:
 ; CHECK: call void @f_readonly() [ "foo"(i32* %x) ]
  call void @f_readonly() [ "foo"(i32* %x) ]
  ret void
}

define void @test_1(i32* %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_1(i32* %x) #4 {
 entry:
 ; CHECK: call void @f_readnone() [ "foo"(i32* %x) ]
  call void @f_readnone() [ "foo"(i32* %x) ]
  ret void
}

define void @test_2(i32* %x) {
; FunctionAttrs must not infer writeonly

; CHECK-LABEL: define void @test_2(i32* %x) {
 entry:
 ; CHECK: call void @f_writeonly() [ "foo"(i32* %x) ]
  call void @f_writeonly() [ "foo"(i32* %x) ]
  ret void
}

define void @test_3(i32* %x) {
; The "deopt" operand bundle does not capture or write to %x.

; CHECK-LABEL: define void @test_3(i32* nocapture readonly %x)
 entry:
  call void @f_readonly() [ "deopt"(i32* %x) ]
  ret void
}

; CHECK: attributes #0 = { nofree readonly }
; CHECK: attributes #1 = { nofree nosync readnone }
; CHECK: attributes #2 = { writeonly }
; CHECK: attributes #3 = { nofree }
; CHECK: attributes #4 = { nofree nosync }

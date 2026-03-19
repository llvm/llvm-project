; RUN: opt -S -passes=inferattrs,function-attrs < %s | FileCheck %s

declare void @f_readonly() readonly
declare void @f_readnone() readnone
declare void @f_writeonly() writeonly

define void @test_0(ptr %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_0(ptr %x) #3 {
 entry:
 ; CHECK: call void @f_readonly() [ "foo"(ptr %x) ]
  call void @f_readonly() [ "foo"(ptr %x) ]
  ret void
}

define void @test_1(ptr %x) {
; FunctionAttrs must not infer readonly / readnone for %x

; CHECK-LABEL: define void @test_1(ptr %x) #4 {
 entry:
 ; CHECK: call void @f_readnone() [ "foo"(ptr %x) ]
  call void @f_readnone() [ "foo"(ptr %x) ]
  ret void
}

define void @test_2(ptr %x) {
; FunctionAttrs must not infer writeonly

; CHECK-LABEL: define void @test_2(ptr %x) {
 entry:
 ; CHECK: call void @f_writeonly() [ "foo"(ptr %x) ]
  call void @f_writeonly() [ "foo"(ptr %x) ]
  ret void
}

define void @test_3(ptr %x) {
; The "deopt" operand bundle does not capture or write to %x.

; CHECK-LABEL: define void @test_3(ptr readonly captures(none) %x)
 entry:
  call void @f_readonly() [ "deopt"(ptr %x) ]
  ret void
}

; CHECK: attributes #0 = { nofree memory(read) }
; CHECK: attributes #1 = { nofree nosync memory(none) }
; CHECK: attributes #2 = { memory(write) }
; CHECK: attributes #3 = { nofree }
; CHECK: attributes #4 = { nofree nosync }

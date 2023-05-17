; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

@x = global i32 0

define void @test_opt(ptr %p) {
; CHECK-LABEL: @test_opt
; CHECK: (ptr nocapture readnone %p) #0 {
  ret void
}

define void @test_optnone(ptr %p) noinline optnone {
; CHECK-LABEL: @test_optnone
; CHECK: (ptr %p) #1 {
  ret void
}

declare i8 @strlen(ptr) noinline optnone
; CHECK-LABEL: @strlen
; CHECK: (ptr) #1

; CHECK-LABEL: attributes #0
; CHECK: = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
; CHECK-LABEL: attributes #1
; CHECK: = { noinline optnone }

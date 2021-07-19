; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

@x = global i32 0

define void @test_opt(i8* %p) {
; CHECK-LABEL: @test_opt
; CHECK: (i8* nocapture readnone %p) #0 {
  ret void
}

define void @test_optnone(i8* %p) noinline optnone {
; CHECK-LABEL: @test_optnone
; CHECK: (i8* %p) #1 {
  ret void
}

declare i8 @strlen(i8*) noinline optnone
; CHECK-LABEL: @strlen
; CHECK: (i8*) #1

; CHECK-LABEL: attributes #0
; CHECK: = { mustprogress nofree norecurse nosync nounwind readnone willreturn }
; CHECK-LABEL: attributes #1
; CHECK: = { noinline optnone }

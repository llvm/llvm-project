; RUN: opt -objc-arc-expand -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @objc_retain(i8*)
declare i8* @objc_autorelease(i8*)

declare void @use_pointer(i8*)

; CHECK: define void @test0
; CHECK: call void @use_pointer(i8* %x)
; CHECK: }
define void @test0(i8* %x) nounwind {
entry:
  %0 = call i8* @objc_retain(i8* %x) nounwind
  call void @use_pointer(i8* %0)
  ret void
}

; CHECK: define void @test1
; CHECK: call void @use_pointer(i8* %x)
; CHECK: }
define void @test1(i8* %x) nounwind {
entry:
  %0 = call i8* @objc_autorelease(i8* %x) nounwind
  call void @use_pointer(i8* %x)
  ret void
}

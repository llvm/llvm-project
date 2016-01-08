; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "x86_64-pc-windows-msvc"

declare void @throw()

declare i32 @__CxxFrameHandler3(...)

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca i8*, align 4
  %alloca1 = alloca i8*, align 4
  store volatile i8* null, i8** %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; CHECK-LABEL: test1:
; CHECK: movq  $0, -16(%rbp)
; CHECK: callq throw

catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 0, i8** %alloca1]
  store volatile i8* null, i8** %alloca1
  %bc1 = bitcast i8** %alloca1 to i8*
  call void @llvm.lifetime.end(i64 4, i8* nonnull %bc1)
  %bc2 = bitcast i8** %alloca2 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %bc2)
  store volatile i8* null, i8** %alloca1
  unreachable

; CHECK-LABEL: "?catch$2@?0?test1@4HA"
; CHECK: movq  $0, -16(%rbp)
; CHECK: movq  $0, -16(%rbp)
; CHECK: ud2

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK-LABEL: $cppxdata$test1:
; CHECK: .long   32                      # CatchObjOffset

define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca i8*, align 4
  %alloca1 = alloca i8*, align 4
  store volatile i8* null, i8** %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; CHECK-LABEL: test2:
; CHECK: movq  $0, -16(%rbp)
; CHECK: callq throw

catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 0, i8** null]
  store volatile i8* null, i8** %alloca1
  %bc1 = bitcast i8** %alloca1 to i8*
  call void @llvm.lifetime.end(i64 4, i8* nonnull %bc1)
  %bc2 = bitcast i8** %alloca2 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %bc2)
  store volatile i8* null, i8** %alloca1
  unreachable

; CHECK-LABEL: "?catch$2@?0?test2@4HA"
; CHECK: movq  $0, -16(%rbp)
; CHECK: movq  $0, -16(%rbp)
; CHECK: ud2

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK-LABEL: $cppxdata$test2:
; CHECK: .long   0                       # CatchObjOffset


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

attributes #0 = { argmemonly nounwind }

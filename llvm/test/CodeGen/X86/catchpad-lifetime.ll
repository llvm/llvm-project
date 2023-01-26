; RUN: llc -opaque-pointers=0 -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: llc -opaque-pointers=0 -mtriple=i686-windows-msvc < %s | FileCheck %s --check-prefix=X86

declare void @throw()

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.trap()

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca i8*, align 4
  %alloca1 = alloca i8*, align 4
  store volatile i8* null, i8** %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; X64-LABEL: test1:
; X64: movq  $0, -8(%rbp)
; X64: callq throw

; X86-LABEL: _test1:
; X86: pushl   %ebp
; X86: movl    %esp, %ebp
; X86: pushl   %ebx
; X86: pushl   %edi
; X86: pushl   %esi
; X86: subl    $24, %esp

; X86: movl  $0, -32(%ebp)
; X86: calll _throw

catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 0, i8** %alloca1]
  %v = load volatile i8*, i8** %alloca1
  store volatile i8* null, i8** %alloca1
  %bc1 = bitcast i8** %alloca1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %bc1)
  %bc2 = bitcast i8** %alloca2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %bc2)
  store volatile i8* null, i8** %alloca1
  call void @llvm.trap()
  unreachable

; X64-LABEL: "?catch$2@?0?test1@4HA"
; X64: movq  $0, -8(%rbp)
; X64: movq  $0, -8(%rbp)
; X64: ud2

; X86-LABEL: "?catch$2@?0?test1@4HA"
; X86: movl  $0, -32(%ebp)
; X86: movl  $0, -32(%ebp)
; X86: ud2

unreachable:                                      ; preds = %entry
  unreachable
}

; X64-LABEL: $cppxdata$test1:
; X64: .long   56                      # CatchObjOffset

; -20 is difference between the end of the EH reg node stack object and the
; catch object at EBP -32.
; X86-LABEL: L__ehtable$test1:
; X86: .long   -20                      # CatchObjOffset

define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca i8*, align 4
  %alloca1 = alloca i8*, align 4
  store volatile i8* null, i8** %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; X64-LABEL: test2:
; X64: movq  $0, -16(%rbp)
; X64: callq throw

; X86-LABEL: _test2:
; X86: movl  $0, -32(%ebp)
; X86: calll _throw


catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 0, i8** null]
  store volatile i8* null, i8** %alloca1
  %bc1 = bitcast i8** %alloca1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %bc1)
  %bc2 = bitcast i8** %alloca2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %bc2)
  store volatile i8* null, i8** %alloca1
  call void @llvm.trap()
  unreachable

; X64-LABEL: "?catch$2@?0?test2@4HA"
; X64: movq  $0, -16(%rbp)
; X64: movq  $0, -16(%rbp)
; X64: ud2

; X86-LABEL: "?catch$2@?0?test2@4HA"
; X86: movl  $0, -32(%ebp)
; X86: movl  $0, -32(%ebp)
; X86: ud2


unreachable:                                      ; preds = %entry
  unreachable
}

; X64-LABEL: $cppxdata$test2:
; X64: .long   0                       # CatchObjOffset


; X86-LABEL: L__ehtable$test2:
; X86: .long   0                       # CatchObjOffset


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

attributes #0 = { argmemonly nounwind }

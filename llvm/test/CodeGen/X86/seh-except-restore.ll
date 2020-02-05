; RUN: llc < %s | FileCheck %s

; In PR44697, the register allocator inserted loads into the __except block
; before the instructions that restore EBP and ESP back to what they should be.
; Make sure they are the first instructions in the __except block.

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.24.28315"

declare i8* @llvm.frameaddress.p0i8(i32 immarg)
declare i8* @llvm.eh.recoverfp(i8*, i8*)
declare i8* @llvm.localrecover(i8*, i8*, i32 immarg)
declare dso_local i32 @_except_handler3(...)
declare void @llvm.localescape(...)

define dso_local zeroext i1 @invokewrapper(
    void ()* nocapture %Fn,
    i1 zeroext %DumpStackAndCleanup,
    i32* nocapture dereferenceable(4) %RetCode)
        personality i32 (...)* @_except_handler3 {
entry:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(i32* nonnull %__exception_code)
  invoke void %Fn()
          to label %return unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i32 ()* @filter to i8*)]
  catchret from %1 to label %__except

__except:                                         ; preds = %__except.ret
  %2 = load i32, i32* %__exception_code, align 4
  store i32 %2, i32* %RetCode, align 4
  br label %return

return:                                           ; preds = %entry, %__except
  %retval.0 = phi i1 [ false, %__except ], [ true, %entry ]
  ret i1 %retval.0
}

; CHECK-LABEL: _invokewrapper:                         # @invokewrapper
; CHECK:         calll   *8(%ebp)
; CHECK: LBB0_2:                                 # %return

; CHECK: LBB0_1:                                 # %__except.ret
; CHECK-NEXT:         movl    -24(%ebp), %esp
; CHECK-NEXT:         addl    $12, %ebp

; Function Attrs: nofree nounwind
define internal i32 @filter() {
entry:
  %0 = tail call i8* @llvm.frameaddress.p0i8(i32 1)
  %1 = tail call i8* @llvm.eh.recoverfp(i8* bitcast (i1 (void ()*, i1, i32*)* @invokewrapper to i8*), i8* %0)
  %2 = tail call i8* @llvm.localrecover(i8* bitcast (i1 (void ()*, i1, i32*)* @invokewrapper to i8*), i8* %1, i32 0)
  %__exception_code = bitcast i8* %2 to i32*
  %3 = getelementptr inbounds i8, i8* %0, i32 -20
  %4 = bitcast i8* %3 to { i32*, i8* }**
  %5 = load { i32*, i8* }*, { i32*, i8* }** %4, align 4
  %6 = getelementptr inbounds { i32*, i8* }, { i32*, i8* }* %5, i32 0, i32 0
  %7 = load i32*, i32** %6, align 4
  %8 = load i32, i32* %7, align 4
  store i32 %8, i32* %__exception_code, align 4
  ret i32 1
}

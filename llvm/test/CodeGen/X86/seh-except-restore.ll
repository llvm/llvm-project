; RUN: llc < %s | FileCheck %s

; In PR44697, the register allocator inserted loads into the __except block
; before the instructions that restore EBP and ESP back to what they should be.
; Make sure they are the first instructions in the __except block.

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.24.28315"

declare ptr @llvm.frameaddress.p0(i32 immarg)
declare ptr @llvm.eh.recoverfp(ptr, ptr)
declare ptr @llvm.localrecover(ptr, ptr, i32 immarg)
declare dso_local i32 @_except_handler3(...)
declare void @llvm.localescape(...)

define dso_local zeroext i1 @invokewrapper(
    ptr nocapture %Fn,
    i1 zeroext %DumpStackAndCleanup,
    ptr nocapture dereferenceable(4) %RetCode)
        personality ptr @_except_handler3 {
entry:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr nonnull %__exception_code)
  invoke void %Fn()
          to label %return unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @filter]
  catchret from %1 to label %__except

__except:                                         ; preds = %__except.ret
  %2 = load i32, ptr %__exception_code, align 4
  store i32 %2, ptr %RetCode, align 4
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
  %0 = tail call ptr @llvm.frameaddress.p0(i32 1)
  %1 = tail call ptr @llvm.eh.recoverfp(ptr @invokewrapper, ptr %0)
  %2 = tail call ptr @llvm.localrecover(ptr @invokewrapper, ptr %1, i32 0)
  %3 = getelementptr inbounds i8, ptr %0, i32 -20
  %4 = load ptr, ptr %3, align 4
  %5 = getelementptr inbounds { ptr, ptr }, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 4
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  ret i32 1
}

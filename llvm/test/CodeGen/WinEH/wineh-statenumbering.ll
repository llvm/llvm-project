; RUN: opt -mtriple=i686-pc-windows-msvc -S -x86-winehstate  < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }
%eh.CatchableType = type { i32, ptr, i32, i32, i32, i32, ptr }
%eh.CatchableTypeArray.1 = type { i32, [1 x ptr] }
%eh.ThrowInfo = type { i32, ptr, ptr, ptr }

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"\01??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, ptr @"\01??_R0H@8", i32 0, i32 -1, i32 0, i32 4, ptr null }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x ptr] [ptr @"_CT??_R0H@84"] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, ptr null, ptr null, ptr @_CTA1H }, section ".xdata", comdat

define i32 @main() #0 personality ptr @__CxxFrameHandler3 {
entry:
  %tmp = alloca i32, align 4
  ; CHECK: entry:
  ; CHECK:   store i32 -1
  ; CHECK:   call void @g(i32 3)
  ; CHECK-NEXT:   call void @g(i32 4)
  ; CHECK-NEXT:   call void @g(i32 5)
  call void @g(i32 3)
  call void @g(i32 4)
  call void @g(i32 5)
  store i32 0, ptr %tmp, align 4
  ; CHECK:   store i32 0
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(ptr %tmp, ptr nonnull @_TI1H) #1
          to label %unreachable.for.entry unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr null, i32 u0x40, ptr null]
  ; CHECK: catch:
  ; CHECK:   store i32 2
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(ptr null, ptr null) [ "funclet"(token %0) ]
          to label %unreachable unwind label %catch.dispatch.1

catch.dispatch.1:                                 ; preds = %catch
  %cs2 = catchswitch within %0 [label %catch.3] unwind to caller
catch.3:                                          ; preds = %catch.dispatch.1
  %1 = catchpad within %cs2 [ptr null, i32 u0x40, ptr null]
  ; CHECK: catch.3:
  ; CHECK:   store i32 3
  ; CHECK:   call void @g(i32 1)
  ; CHECK-NEXT:   call void @g(i32 2)
  ; CHECK-NEXT:   call void @g(i32 3)
  call void @g(i32 1)
  call void @g(i32 2)
  call void @g(i32 3)
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.3
  ; CHECK: try.cont:
  ; CHECK:   store i32 1
  ; CHECK:   call void @g(i32 2)
  ; CHECK-NEXT:   call void @g(i32 3)
  ; CHECK-NEXT:   call void @g(i32 4)
  call void @g(i32 2)
  call void @g(i32 3)
  call void @g(i32 4)
  unreachable

unreachable:                                      ; preds = %catch
  unreachable

unreachable.for.entry:                            ; preds = %entry
  unreachable
}

define i32 @nopads() #0 personality ptr @__CxxFrameHandler3 {
  ret i32 0
}

; CHECK-LABEL: define i32 @nopads()
; CHECK-NEXT: ret i32 0
; CHECK-NOT: __ehhandler$nopads

; CHECK-LABEL: define void @PR25926()
define void @PR25926() personality ptr @__CxxFrameHandler3 {
entry:
  ; CHECK: entry:
  ; CHECK:   store i32 -1
  ; CHECK:   store i32 0
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(ptr null, ptr null)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  ; CHECK: catch:
  ; CHECK:   store i32 3
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(ptr null, ptr null) [ "funclet"(token %1) ]
          to label %unreachable1 unwind label %catch.dispatch1

catch.dispatch1:                                  ; preds = %catch
  %2 = catchswitch within %1 [label %catch2] unwind label %ehcleanup

catch2:                                           ; preds = %catch.dispatch1
  %3 = catchpad within %2 [ptr null, i32 64, ptr null]
  catchret from %3 to label %try.cont

try.cont:                                         ; preds = %catch2
  ; CHECK: try.cont:
  ; CHECK:   store i32 1
  ; CHECK:   call void @dtor()
  ; CHECK-NEXT:   call void @dtor()
  ; CHECK-NEXT:   call void @dtor()
  call void @dtor() #3 [ "funclet"(token %1) ]
  call void @dtor() #3 [ "funclet"(token %1) ]
  call void @dtor() #3 [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont4

try.cont4:                                        ; preds = %try.cont
  ret void

ehcleanup:                                        ; preds = %catch.dispatch1
  %4 = cleanuppad within %1 []
  ; CHECK: ehcleanup:
  ; CHECK:   call void @dtor()
  call void @dtor() #3 [ "funclet"(token %4) ]
  cleanupret from %4 unwind to caller

unreachable:                                      ; preds = %entry
  unreachable

unreachable1:                                     ; preds = %catch
  unreachable
}

; CHECK-LABEL: define void @required_state_store(
define void @required_state_store(i1 zeroext %cond) personality ptr @_except_handler3 {
entry:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr nonnull %__exception_code)
; CHECK:   store i32 -1
; CHECK:   call void @g(i32 0)
  call void @g(i32 0)
  br i1 %cond, label %if.then, label %if.end

if.then:                                          ; preds = %entry
; CHECK:        store i32 0
; CHECK-NEXT:   invoke void @g(i32 1)
  invoke void @g(i32 1)
          to label %if.end unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.then
  %0 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @"\01?filt$0@0@required_state_store@@"]
  catchret from %1 to label %if.end

if.end:                                           ; preds = %if.then, %__except.ret, %entry
; CHECK:        store i32 -1
; CHECK-NEXT:   call void @dtor()
  call void @dtor()
  ret void
}

define internal i32 @"\01?filt$0@0@required_state_store@@"() {
entry:
  %0 = tail call ptr @llvm.frameaddress(i32 1)
  %1 = tail call ptr @llvm.eh.recoverfp(ptr @required_state_store, ptr %0)
  %2 = tail call ptr @llvm.localrecover(ptr @required_state_store, ptr %1, i32 0)
  %3 = getelementptr inbounds i8, ptr %0, i32 -20
  %4 = load ptr, ptr %3, align 4
  %5 = getelementptr inbounds { ptr, ptr }, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 4
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  ret i32 1
}

declare void @g(i32) #0

declare void @dtor()

declare x86_stdcallcc void @_CxxThrowException(ptr, ptr)

declare i32 @__CxxFrameHandler3(...)

declare ptr @llvm.frameaddress(i32)

declare ptr @llvm.eh.recoverfp(ptr, ptr)

declare ptr @llvm.localrecover(ptr, ptr, i32)

declare void @llvm.localescape(...)

declare i32 @_except_handler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 245153) (llvm/trunk 245238)"}

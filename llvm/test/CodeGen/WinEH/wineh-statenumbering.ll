; RUN: opt -mtriple=i686-pc-windows-msvc -S -x86-winehstate  < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i8*, i32, i32, i32, i32, i8* }
%eh.CatchableTypeArray.1 = type { i32, [1 x %eh.CatchableType*] }
%eh.ThrowInfo = type { i32, i8*, i8*, i8* }

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i32 -1, i32 0, i32 4, i8* null }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0H@84"] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i8* null, i8* null, i8* bitcast (%eh.CatchableTypeArray.1* @_CTA1H to i8*) }, section ".xdata", comdat

define i32 @main() #0 personality i32 (...)* @__CxxFrameHandler3 {
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
  store i32 0, i32* %tmp, align 4
  %0 = bitcast i32* %tmp to i8*
  ; CHECK:   store i32 0
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* nonnull @_TI1H) #1
          to label %unreachable.for.entry unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  ; CHECK: catch:
  ; CHECK:   store i32 2
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) [ "funclet"(token %1) ]
          to label %unreachable unwind label %catch.dispatch.1

catch.dispatch.1:                                 ; preds = %catch
  %cs2 = catchswitch within %1 [label %catch.3] unwind to caller
catch.3:                                          ; preds = %catch.dispatch.1
  %2 = catchpad within %cs2 [i8* null, i32 u0x40, i8* null]
  ; CHECK: catch.3:
  ; CHECK:   store i32 3
  ; CHECK:   call void @g(i32 1)
  ; CHECK-NEXT:   call void @g(i32 2)
  ; CHECK-NEXT:   call void @g(i32 3)
  call void @g(i32 1)
  call void @g(i32 2)
  call void @g(i32 3)
  catchret from %2 to label %try.cont

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

define i32 @nopads() #0 personality i32 (...)* @__CxxFrameHandler3 {
  ret i32 0
}

; CHECK-LABEL: define i32 @nopads()
; CHECK-NEXT: ret i32 0
; CHECK-NOT: __ehhandler$nopads

; CHECK-LABEL: define void @PR25926()
define void @PR25926() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  ; CHECK: entry:
  ; CHECK:   store i32 -1
  ; CHECK:   store i32 0
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  ; CHECK: catch:
  ; CHECK:   store i32 3
  ; CHECK:   invoke void @_CxxThrowException(
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) [ "funclet"(token %1) ]
          to label %unreachable1 unwind label %catch.dispatch1

catch.dispatch1:                                  ; preds = %catch
  %2 = catchswitch within %1 [label %catch2] unwind label %ehcleanup

catch2:                                           ; preds = %catch.dispatch1
  %3 = catchpad within %2 [i8* null, i32 64, i8* null]
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
define void @required_state_store(i1 zeroext %cond) personality i32 (...)* @_except_handler3 {
entry:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(i32* nonnull %__exception_code)
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
  %1 = catchpad within %0 [i8* bitcast (i32 ()* @"\01?filt$0@0@required_state_store@@" to i8*)]
  catchret from %1 to label %if.end

if.end:                                           ; preds = %if.then, %__except.ret, %entry
; CHECK:        store i32 -1
; CHECK-NEXT:   call void @dtor()
  call void @dtor()
  ret void
}

define internal i32 @"\01?filt$0@0@required_state_store@@"() {
entry:
  %0 = tail call i8* @llvm.frameaddress(i32 1)
  %1 = tail call i8* @llvm.eh.recoverfp(i8* bitcast (void (i1)* @required_state_store to i8*), i8* %0)
  %2 = tail call i8* @llvm.localrecover(i8* bitcast (void (i1)* @required_state_store to i8*), i8* %1, i32 0)
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

declare void @g(i32) #0

declare void @dtor()

declare x86_stdcallcc void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

declare i8* @llvm.frameaddress(i32)

declare i8* @llvm.eh.recoverfp(i8*, i8*)

declare i8* @llvm.localrecover(i8*, i8*, i32)

declare void @llvm.localescape(...)

declare i32 @_except_handler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 245153) (llvm/trunk 245238)"}

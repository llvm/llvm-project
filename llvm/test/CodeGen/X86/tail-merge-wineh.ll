; RUN: llc < %s | FileCheck %s

; Started from this code:
; void f() {
;   try {
;     try {
;       throw 42;
;     } catch (int) {
;     }
;     try {
;       throw 42;
;     } catch (int) {
;     }
;   } catch (int) {
;   }
; }

; Don't tail merge the calls.
; CHECK: calll __CxxThrowException@8
; CHECK: calll __CxxThrowException@8

; ModuleID = 'cppeh-pingpong.cpp'
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

define void @"\01?f@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %i = alloca i32, align 4
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  store i32 0, i32* %i, align 4
  store i32 42, i32* %tmp, align 4
  %0 = bitcast i32* %tmp to i8*
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind label %catch.dispatch.7

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest
  store i32 42, i32* %tmp1, align 4
  %2 = bitcast i32* %tmp1 to i8*
  invoke void @_CxxThrowException(i8* %2, %eh.ThrowInfo* @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch.2

catch.dispatch.2:                                 ; preds = %try.cont
  %cs2 = catchswitch within none [label %catch.4] unwind label %catch.dispatch.7

catch.4:                                          ; preds = %catch.dispatch.2
  %3 = catchpad within %cs2 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  catchret from %3 to label %catchret.dest.5

catchret.dest.5:                                  ; preds = %catch.4
  br label %try.cont.6

try.cont.6:                                       ; preds = %catchret.dest.5
  br label %try.cont.11

catch.dispatch.7:
  %cs3 = catchswitch within none [label %catch.9] unwind to caller

catch.9:                                          ; preds = %catch.dispatch.7
  %4 = catchpad within %cs3 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  catchret from %4 to label %catchret.dest.10

catchret.dest.10:                                 ; preds = %catch.9
  br label %try.cont.11

try.cont.11:                                      ; preds = %catchret.dest.10, %try.cont.6
  ret void

unreachable:                                      ; preds = %try.cont, %entry
  unreachable
}

declare x86_stdcallcc void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

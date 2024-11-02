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

define void @"\01?f@@YAXXZ"() #0 personality ptr @__CxxFrameHandler3 {
entry:
  %i = alloca i32, align 4
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  store i32 0, ptr %i, align 4
  store i32 42, ptr %tmp, align 4
  invoke void @_CxxThrowException(ptr %tmp, ptr @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind label %catch.dispatch.7

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr @"\01??_R0H@8", i32 0, ptr null]
  catchret from %0 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest
  store i32 42, ptr %tmp1, align 4
  invoke void @_CxxThrowException(ptr %tmp1, ptr @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch.2

catch.dispatch.2:                                 ; preds = %try.cont
  %cs2 = catchswitch within none [label %catch.4] unwind label %catch.dispatch.7

catch.4:                                          ; preds = %catch.dispatch.2
  %1 = catchpad within %cs2 [ptr @"\01??_R0H@8", i32 0, ptr null]
  catchret from %1 to label %catchret.dest.5

catchret.dest.5:                                  ; preds = %catch.4
  br label %try.cont.6

try.cont.6:                                       ; preds = %catchret.dest.5
  br label %try.cont.11

catch.dispatch.7:
  %cs3 = catchswitch within none [label %catch.9] unwind to caller

catch.9:                                          ; preds = %catch.dispatch.7
  %2 = catchpad within %cs3 [ptr @"\01??_R0H@8", i32 0, ptr null]
  catchret from %2 to label %catchret.dest.10

catchret.dest.10:                                 ; preds = %catch.9
  br label %try.cont.11

try.cont.11:                                      ; preds = %catchret.dest.10, %try.cont.6
  ret void

unreachable:                                      ; preds = %try.cont, %entry
  unreachable
}

declare x86_stdcallcc void @_CxxThrowException(ptr, ptr)

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

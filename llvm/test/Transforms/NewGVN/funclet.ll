; RUN: opt -passes=newgvn -S < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%eh.ThrowInfo = type { i32, i8*, i8*, i8* }
%struct.A = type { i32* }

@"_TI1?AUA@@" = external constant %eh.ThrowInfo

define i8 @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %b = alloca i8
  %c = alloca i8
  store i8 42, i8* %b
  store i8 13, i8* %c
  invoke void @_CxxThrowException(i8* %b, %eh.ThrowInfo* nonnull @"_TI1?AUA@@")
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %catchpad = catchpad within %cs1 [i8* null, i32 64, i8* null]
  store i8 5, i8* %b
  catchret from %catchpad to label %try.cont

try.cont:                                         ; preds = %catch
  %load_b = load i8, i8* %b
  %load_c = load i8, i8* %c
  %add = add i8 %load_b, %load_c
  ret i8 %add

unreachable:                                      ; preds = %entry
  unreachable
}
; CHECK-LABEL: define i8 @f(
; CHECK:       %[[load_b:.*]] = load i8, i8* %b
; CHECK-NEXT:  %[[load_c:.*]] = load i8, i8* %c
; CHECK-NEXT:  %[[add:.*]] = add i8 %[[load_b]], %[[load_c]]
; CHECK-NEXT:  ret i8 %[[add]]

declare i32 @__CxxFrameHandler3(...)

declare x86_stdcallcc void @_CxxThrowException(i8*, %eh.ThrowInfo*)

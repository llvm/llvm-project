; RUN: opt -passes=gvn -S < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

%eh.ThrowInfo = type { i32, ptr, ptr, ptr }
%struct.A = type { ptr }

@"_TI1?AUA@@" = external constant %eh.ThrowInfo

define i8 @f() personality ptr @__CxxFrameHandler3 {
entry:
  %b = alloca i8
  %c = alloca i8
  store i8 42, ptr %b
  store i8 13, ptr %c
  invoke void @_CxxThrowException(ptr %b, ptr nonnull @"_TI1?AUA@@")
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %catchpad = catchpad within %cs1 [ptr null, i32 64, ptr null]
  store i8 5, ptr %b
  catchret from %catchpad to label %try.cont

try.cont:                                         ; preds = %catch
  %load_b = load i8, ptr %b
  %load_c = load i8, ptr %c
  %add = add i8 %load_b, %load_c
  ret i8 %add

unreachable:                                      ; preds = %entry
  unreachable
}
; CHECK-LABEL: define i8 @f(
; CHECK:       %[[load_b:.*]] = load i8, ptr %b
; CHECK-NEXT:  %[[load_c:.*]] = load i8, ptr %c
; CHECK-NEXT:  %[[add:.*]] = add i8 %[[load_b]], %[[load_c]]
; CHECK-NEXT:  ret i8 %[[add]]

declare i32 @__CxxFrameHandler3(...)

declare x86_stdcallcc void @_CxxThrowException(ptr, ptr)

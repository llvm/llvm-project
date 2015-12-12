; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%eh.ThrowInfo = type { i32, i32, i32, i32 }
%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = internal global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }

define void @test1(i1 %B) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs1 [i8* null, i32 64, i8* null]
  br label %catch.loop

catch.loop:
  br i1 %B, label %catchret, label %catch.loop

catchret:
  catchret from %cp to label %try.cont

try.cont:
  ret void

unreachable:
  unreachable
}

; CHECK-LABEL: test1:

; The entry funclet contains %entry and %try.cont
; CHECK: # %entry
; CHECK: # %try.cont
; CHECK: retq

; The catch funclet contains %catch and %catchret
; CHECK: # %catch{{$}}
; CHECK: # %catchret
; CHECK: retq

declare void @g()


define i32 @test2(i1 %B) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #1
          to label %unreachable unwind label %catch.dispatch.1

catch.dispatch.1:                                 ; preds = %catch
  %cs2 = catchswitch within %0 [label %catch.3] unwind to caller

catch.3:                                          ; preds = %catch.dispatch.1
  %1 = catchpad within %cs2 [i8* null, i32 64, i8* null]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.3
  catchret from %0 to label %try.cont.5

try.cont.5:                                       ; preds = %try.cont
  ret i32 0

unreachable:                                      ; preds = %catch, %entry
  unreachable
}

; CHECK-LABEL: test2:

; The parent function contains %entry and %try.cont.5
; CHECK: .seh_proc
; CHECK: # %entry
; CHECK: # %try.cont.5
; CHECK: retq

; The inner catch funclet contains %catch.3
; CHECK: .seh_proc
; CHECK: # %catch.3{{$}}
; CHECK: retq

; The outer catch funclet contains %catch
; CHECK: .seh_proc
; CHECK: # %catch{{$}}
; CHECK: callq _CxxThrowException
; CHECK: # %unreachable
; CHECK: ud2


define void @test3(i1 %V) #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch.2] unwind label %catch.dispatch.1

catch.2:                                          ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  tail call void @exit(i32 0) #2
  unreachable

catch.dispatch.1:                                 ; preds = %catch.dispatch
  %cs2 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch.1
  %1 = catchpad within %cs2 [i8* null, i32 64, i8* null]
  tail call void @exit(i32 0) #2
  unreachable

try.cont:                                         ; preds = %entry
  br i1 %V, label %exit_one, label %exit_two

exit_one:
  tail call void @exit(i32 0)
  unreachable

exit_two:
  tail call void @exit(i32 0)
  unreachable
}

; CHECK-LABEL: test3:

; The entry funclet contains %entry and %try.cont
; CHECK: # %entry
; CHECK: # %try.cont
; CHECK: callq exit
; CHECK-NOT: # exit_one
; CHECK-NOT: # exit_two
; CHECK: ud2

; The catch(...) funclet contains %catch.2
; CHECK: # %catch.2{{$}}
; CHECK: callq exit
; CHECK: ud2

; The catch(int) funclet contains %catch
; CHECK: # %catch{{$}}
; CHECK: callq exit
; CHECK: ud2

declare void @exit(i32) noreturn nounwind
declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)
declare i32 @__CxxFrameHandler3(...)

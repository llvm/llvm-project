; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

declare i32 @_except_handler3(...)
declare i32 @__CxxFrameHandler3(...)

declare void @external(i32*)
declare void @reserve()

define void @f() personality i32 (...)* @_except_handler3 {
entry:
  br label %throw

throw:                                            ; preds = %throw, %entry
  %tmp96 = getelementptr inbounds i8, i8* undef, i32 1
  invoke void @reserve()
          to label %throw unwind label %pad

pad:                                              ; preds = %throw
  %phi2 = phi i8* [ %tmp96, %throw ]
  %cs = catchswitch within none [label %unreachable] unwind label %blah2

unreachable:
  catchpad within %cs []
  unreachable

blah2:
  %cleanuppadi4.i.i.i = cleanuppad within none []
  br label %loop_body

loop_body:                                        ; preds = %iter, %pad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %blah2 ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body

unwind_out:                                       ; preds = %iter, %loop_body
  cleanupret from %cleanuppadi4.i.i.i unwind to caller
}

; CHECK-LABEL: define void @f(
; CHECK: cleanuppad within none []
; CHECK-NEXT: ptrtoint i8* %phi2 to i32

define void @g() personality i32 (...)* @_except_handler3 {
entry:
  br label %throw

throw:                                            ; preds = %throw, %entry
  %tmp96 = getelementptr inbounds i8, i8* undef, i32 1
  invoke void @reserve()
          to label %throw unwind label %pad

pad:
  %phi2 = phi i8* [ %tmp96, %throw ]
  %cs = catchswitch within none [label %unreachable, label %blah] unwind to caller

unreachable:
  catchpad within %cs []
  unreachable

blah:
  %catchpad = catchpad within %cs []
  br label %loop_body

unwind_out:
  catchret from %catchpad to label %leave

leave:
  ret void

loop_body:                                        ; preds = %iter, %pad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %blah ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body
}

; CHECK-LABEL: define void @g(
; CHECK: blah:
; CHECK-NEXT: catchpad within %cs []
; CHECK-NEXT: ptrtoint i8* %phi2 to i32


define void @h() personality i32 (...)* @_except_handler3 {
entry:
  br label %throw

throw:                                            ; preds = %throw, %entry
  %tmp96 = getelementptr inbounds i8, i8* undef, i32 1
  invoke void @reserve()
          to label %throw unwind label %pad

pad:
  %cs = catchswitch within none [label %unreachable, label %blug] unwind to caller

unreachable:
  catchpad within %cs []
  unreachable

blug:
  %phi2 = phi i8* [ %tmp96, %pad ]
  %catchpad = catchpad within %cs []
  br label %loop_body

unwind_out:
  catchret from %catchpad to label %leave

leave:
  ret void

loop_body:                                        ; preds = %iter, %pad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %blug ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body
}

; CHECK-LABEL: define void @h(
; CHECK: blug:
; CHECK: catchpad within %cs []
; CHECK-NEXT: ptrtoint i8* %phi2 to i32

define void @i() personality i32 (...)* @_except_handler3 {
entry:
  br label %throw

throw:                                            ; preds = %throw, %entry
  %tmp96 = getelementptr inbounds i8, i8* undef, i32 1
  invoke void @reserve()
          to label %throw unwind label %catchpad

catchpad:                                              ; preds = %throw
  %phi2 = phi i8* [ %tmp96, %throw ]
  %cs = catchswitch within none [label %cp_body] unwind label %cleanuppad

cp_body:
  catchpad within %cs []
  br label %loop_head

cleanuppad:
  cleanuppad within none []
  br label %loop_head

loop_head:
  br label %loop_body

loop_body:                                        ; preds = %iter, %catchpad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %loop_head ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body

unwind_out:                                       ; preds = %iter, %loop_body
  unreachable
}

; CHECK-LABEL: define void @i(
; CHECK: ptrtoint i8* %phi2 to i32

define void @test1(i32* %b, i32* %c) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %d.0 = phi i32* [ %b, %entry ], [ %incdec.ptr, %for.inc ]
  invoke void @external(i32* %d.0)
          to label %for.inc unwind label %catch.dispatch

for.inc:                                          ; preds = %for.cond
  %incdec.ptr = getelementptr inbounds i32, i32* %d.0, i32 1
  br label %for.cond

catch.dispatch:                                   ; preds = %for.cond
  %cs = catchswitch within none [label %catch] unwind label %catch.dispatch.2

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs [i8* null, i32 64, i8* null]
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %catch
  invoke void @external(i32* %c)
          to label %try.cont.7 unwind label %catch.dispatch.2

catch.dispatch.2:                                 ; preds = %try.cont, %catchendblock
  %e.0 = phi i32* [ %c, %try.cont ], [ %b, %catch.dispatch ]
  %cs2 = catchswitch within none [label %catch.4] unwind to caller

catch.4:                                          ; preds = %catch.dispatch.2
  catchpad within %cs2 [i8* null, i32 64, i8* null]
  unreachable

try.cont.7:                                       ; preds = %try.cont
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK: for.cond:
; CHECK:   %d.0 = phi i32* [ %b, %entry ], [ %incdec.ptr, %for.inc ]

; CHECK: catch.dispatch.2:
; CHECK: %e.0 = phi i32* [ %c, %try.cont ], [ %b, %catch.dispatch ]

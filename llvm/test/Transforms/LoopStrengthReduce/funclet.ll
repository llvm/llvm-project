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
  terminatepad [] unwind label %blah

blah:
  catchpad [] to label %unreachable unwind label %blah3

unreachable:
  unreachable

blah3:
  catchendpad unwind label %blah2

blah2:
  %cleanuppadi4.i.i.i = cleanuppad []
  br label %loop_body

loop_body:                                        ; preds = %iter, %pad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %blah2 ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body

unwind_out:                                       ; preds = %iter, %loop_body
  cleanupret %cleanuppadi4.i.i.i unwind to caller
}

; CHECK-LABEL: define void @f(
; CHECK: cleanuppad []
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
  catchpad [] to label %unreachable unwind label %blah

unreachable:
  unreachable

blah:
  %catchpad = catchpad [] to label %loop_body unwind label %blah3


blah3:
  catchendpad unwind to caller ;label %blah2

unwind_out:
  catchret %catchpad to label %leave

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
; CHECK-NEXT: catchpad []
; CHECK-NEXT: to label %loop_body.preheader

; CHECK: loop_body.preheader:
; CHECK-NEXT: ptrtoint i8* %phi2 to i32


define void @h() personality i32 (...)* @_except_handler3 {
entry:
  br label %throw

throw:                                            ; preds = %throw, %entry
  %tmp96 = getelementptr inbounds i8, i8* undef, i32 1
  invoke void @reserve()
          to label %throw unwind label %pad

pad:
  catchpad [] to label %unreachable unwind label %blug

unreachable:
  unreachable

blug:
  %phi2 = phi i8* [ %tmp96, %pad ]
  %catchpad = catchpad [] to label %blah2 unwind label %blah3

blah2:
  br label %loop_body

blah3:
  catchendpad unwind to caller ;label %blah2

unwind_out:
  catchret %catchpad to label %leave

leave:
  ret void

loop_body:                                        ; preds = %iter, %pad
  %tmp99 = phi i8* [ %tmp101, %iter ], [ %phi2, %blah2 ]
  %tmp100 = icmp eq i8* %tmp99, undef
  br i1 %tmp100, label %unwind_out, label %iter

iter:                                             ; preds = %loop_body
  %tmp101 = getelementptr inbounds i8, i8* %tmp99, i32 1
  br i1 undef, label %unwind_out, label %loop_body
}

; CHECK-LABEL: define void @h(
; CHECK: blug:
; CHECK: catchpad []
; CHECK-NEXT: to label %blah2

; CHECK: blah2:
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
  catchpad [] to label %cp_body unwind label %catchendpad

cp_body:
  br label %loop_head

catchendpad:
  catchendpad unwind label %cleanuppad

cleanuppad:
  cleanuppad []
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
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catchendblock:                                    ; preds = %catch.dispatch
  catchendpad unwind label %catch.dispatch.2

catch:                                            ; preds = %catch.dispatch
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %catch
  invoke void @external(i32* %c)
          to label %try.cont.7 unwind label %catch.dispatch.2

catch.dispatch.2:                                 ; preds = %try.cont, %catchendblock
  %e.0 = phi i32* [ %c, %try.cont ], [ %b, %catchendblock ]
  %1 = catchpad [i8* null, i32 64, i8* null]
          to label %catch.4 unwind label %catchendblock.3

catch.4:                                          ; preds = %catch.dispatch.2
  unreachable

try.cont.7:                                       ; preds = %try.cont
  ret void

catchendblock.3:                                  ; preds = %catch.dispatch.2
  catchendpad unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: for.cond:
; CHECK:   %d.0 = phi i32* [ %b, %entry ], [ %incdec.ptr, %for.inc ]

; CHECK: catchendpad unwind label %catch.dispatch.2

; CHECK: catch.dispatch.2:
; CHECK: %e.0 = phi i32* [ %c, %try.cont ], [ %b, %catchendblock ]

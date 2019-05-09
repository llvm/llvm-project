; Catchswitch is interesting because reassociate previously tried to insert
; into the catchswitch block, which is impossible.
;
; RUN: opt -reassociate -disable-output < %s
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "catchswitch.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local void @"?f@@YAX_N@Z"(i1 %b) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  invoke void @"?g@@YAXXZ"()
          to label %cleanup unwind label %catch.dispatch

if.else:                                          ; preds = %entry
  invoke void @"?g2@@YAXXZ"()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.else, %if.then
  %z.0 = phi i32 [ 3, %if.then ], [ 5, %if.else ]
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  %blech = sub nsw i32 5, %z.0
  %sub = sub nsw i32 %blech, %z.0
  call void @"?use@@YAXHH@Z"(i32 %z.0, i32 %sub) [ "funclet"(token %1) ]
  unreachable

cleanup:                                          ; preds = %if.else, %if.then
  ret void
}

declare dso_local void @"?g@@YAXXZ"() local_unnamed_addr #0

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @"?g2@@YAXXZ"() local_unnamed_addr #0

declare dso_local void @"?use@@YAXHH@Z"(i32, i32) local_unnamed_addr #0

attributes #0 = { "use-soft-float"="false" }

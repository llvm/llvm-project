; RUN: opt -S -consthoist -consthoist-with-block-frequency=false < %s | FileCheck %s
; RUN: opt -S -consthoist -consthoist-with-block-frequency=true < %s | FileCheck --check-prefix=BFIHOIST %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK-LABEL: define i32 @main
; CHECK: %tobool = icmp eq i32 %argc, 0
; CHECK-NEXT: bitcast i64 9209618997431186100 to i64
; CHECK-NEXT: br i1 %tobool

; BFIHOIST-LABEL: define i32 @main
; BFIHOIST: then:
; BFIHOIST: %[[CONST1:.*]] = bitcast i64 9209618997431186100 to i64
; BFIHOIST: %add = add i64 %call4, %[[CONST1]]
; BFIHOIST: br label %endif
; BFIHOIST: else:
; BFIHOIST: %[[CONST2:.*]] = bitcast i64 9209618997431186100 to i64
; BFIHOIST: %add6 = add i64 %call5, %[[CONST2]]
; BFIHOIST: br label %endif

; Function Attrs: norecurse
define i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  %call = tail call i64 @fn(i64 0)
  %call1 = tail call i64 @fn(i64 1)
  %tobool = icmp eq i32 %argc, 0
  br i1 %tobool, label %2, label %1

; <label>:1:                                      ; preds = %0
  %call2 = invoke i64 @fn(i64 %call)
          to label %6 unwind label %catch.dispatch

; <label>:2:                                      ; preds = %0
  %call3 = invoke i64 @fn(i64 %call1)
          to label %6 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %2, %1
  %z.0 = phi i64 [ %call, %1 ], [ %call1, %2 ]
  %3 = catchswitch within none [label %4] unwind to caller

; <label>:4:                                      ; preds = %catch.dispatch
  %5 = catchpad within %3 [i8* null, i32 64, i8* null]
  br i1 %tobool, label %then, label %else

then:
  %call4 = tail call i64 @fn(i64 %z.0) [ "funclet"(token %5) ]
  %add = add i64 %call4, 9209618997431186100
  br label %endif

else:
  %call5 = tail call i64 @fn(i64 0) [ "funclet"(token %5) ]
  %add6 = add i64 %call5, 9209618997431186100
  br label %endif

endif:
  %v = phi i64 [ %add, %then ], [ %add6, %else ]
  %call7 = tail call i64 @fn(i64 %v) [ "funclet"(token %5) ]
  %call8 = tail call i64 @fn(i64 %call7) [ "funclet"(token %5) ]
  catchret from %5 to label %6

; <label>:6:                                      ; preds = %1, %2, %4
  ret i32 0
}

declare i64 @fn(i64) local_unnamed_addr #1

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { norecurse "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

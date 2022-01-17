; RUN: opt -S -loop-vectorize < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define void @test1() #0 personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @_CxxThrowException(i8* null, i8* null)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  catchret from %1 to label %try.cont

for.body:                                         ; preds = %for.body, %catch
  %i.07 = phi i32 [ 0, %catch ], [ %inc, %for.body ]
  %call = call double @floor(double 1.0) #1 [ "funclet"(token %1) ]
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

try.cont:                                         ; preds = %for.cond.cleanup
  ret void

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK-LABEL: define void @test1(
; CHECK: %[[cpad:.*]] = catchpad within {{.*}} [i8* null, i32 64, i8* null]
; CHECK: call <8 x double> @llvm.floor.v8f64(<8 x double> {{.*}}) [ "funclet"(token %[[cpad]]) ]

declare x86_stdcallcc void @_CxxThrowException(i8*, i8*)

declare i32 @__CxxFrameHandler3(...)

declare double @floor(double) #1

attributes #0 = { "target-features"="+sse2" }
attributes #1 = { nounwind readnone }

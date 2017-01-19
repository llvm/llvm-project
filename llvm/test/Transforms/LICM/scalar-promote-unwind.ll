; RUN: opt < %s -basicaa -licm -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure we don't hoist the store out of the loop; %a would
; have the wrong value if f() unwinds

define void @test1(i32* nocapture noalias %a, i1 zeroext %y) uwtable {
entry:
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  br i1 %y, label %if.then, label %for.inc

; CHECK: define void @test1
; CHECK: load i32, i32*
; CHECK-NEXT: add
; CHECK-NEXT: store i32

if.then:
  tail call void @f()
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; We can hoist the store out of the loop here; if f() unwinds,
; the lifetime of %a ends.

define void @test2(i1 zeroext %y) uwtable {
entry:
  %a = alloca i32
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  br i1 %y, label %if.then, label %for.inc

if.then:
  tail call void @f()
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
; CHECK: define void @test2
; CHECK: store i32
; CHECK-NEXT: ret void
  ret void
}

@_ZTIi = external constant i8*

; In this test, the loop is within a try block. There is an explicit unwind edge out of the loop.
; Make sure this edge is treated as a loop exit, and that the loads and stores are promoted as
; expected
define void @loop_within_tryblock() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %a = alloca i32, align 4
  store i32 0, i32* %a, align 4
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:
; CHECK-NOT: load
; CHECK-NOT: store 
; CHECK: invoke
for.body:
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  invoke void @boo()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

; CHECK: lpad:
; CHECK: store
; CHECK: br
lpad:
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = extractvalue { i8*, i32 } %1, 1
  br label %catch.dispatch

catch.dispatch:
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #3
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %eh.resume

catch:
  %5 = call i8* @__cxa_begin_catch(i8* %2) #3
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  call void @__cxa_end_catch() #3
  br label %try.cont

try.cont:
  ret void

for.end:
  br label %try.cont

eh.resume:
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %2, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %3, 1
  resume { i8*, i32 } %lpad.val3
}

declare void @boo()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare void @f() uwtable

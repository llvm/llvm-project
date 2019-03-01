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

;; We can promote if the load can be proven safe to speculate, and the
;; store safe to sink, even if the the store *isn't* must execute.
define void @test3(i1 zeroext %y) uwtable {
; CHECK-LABEL: @test3
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  %a = alloca i32
; CHECK-NEXT:  %a.promoted = load i32, i32* %a, align 1
  %a = alloca i32
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  tail call void @f()
  store i32 %add, i32* %a, align 4
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
; CHECK-LABEL: for.cond.cleanup:
; CHECK: store i32 %add.lcssa, i32* %a, align 1
; CHECK-NEXT: ret void
  ret void
}

;; Same as test3, but with unordered atomics
;; FIXME: doing the transform w/o alignment here is wrong since we're
;; creating an unaligned atomic which we may not be able to lower.
define void @test3b(i1 zeroext %y) uwtable {
; CHECK-LABEL: @test3
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  %a = alloca i32
; CHECK-NEXT:  %a.promoted = load atomic i32, i32* %a unordered, align 1
  %a = alloca i32
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load atomic i32, i32* %a unordered, align 4
  %add = add nsw i32 %0, 1
  tail call void @f()
  store atomic i32 %add, i32* %a unordered, align 4
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
; CHECK-LABEL: for.cond.cleanup:
; CHECK: store atomic i32 %add.lcssa, i32* %a unordered, align 1
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


; The malloc'ed memory is not capture and therefore promoted.
define void @malloc_no_capture() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = call i8* @malloc(i64 4)
  %0 = bitcast i8* %call to i32*
  br label %for.body

; CHECK: for.body:
; CHECK-NOT: load
; CHECK-NOT: store
; CHECK: br 
for.body:
  %i.0 = phi i32 [ 0, %entry  ], [ %inc, %for.latch ]
  %1 = load i32, i32* %0, align 4
  %add = add nsw i32 %1, 1
  store i32 %add, i32* %0, align 4
  br label %for.call

for.call:
  invoke void @boo()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  br label %for.latch

for.latch:
  %inc = add i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %fun.ret

lpad:
  %2 = landingpad { i8*, i32 }
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  %4 = extractvalue { i8*, i32 } %2, 1
  br label %catch

catch:
  %5 = call i8* @__cxa_begin_catch(i8* %3) #4
  %6 = bitcast i32* %0 to i8*
  call void @free(i8* %6)
  call void @__cxa_end_catch()
  br label %fun.ret

fun.ret:
  ret void
}

; The malloc'ed memory can be captured and therefore not promoted.
define void @malloc_capture(i32** noalias %A) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = call i8* @malloc(i64 4)
  %0 = bitcast i8* %call to i32*
  br label %for.body

; CHECK: for.body:
; CHECK: load
; CHECK: store
; CHECK: br 
for.body:
  %i.0 = phi i32 [ 0, %entry  ], [ %inc, %for.latch ]
  %1 = load i32, i32* %0, align 4
  %add = add nsw i32 %1, 1
  store i32 %add, i32* %0, align 4
  br label %for.call

for.call:
  invoke void @boo_readnone()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  br label %for.latch

for.latch:
  store i32* %0, i32** %A 
  %inc = add i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %fun.ret

lpad:
  %2 = landingpad { i8*, i32 }
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  %4 = extractvalue { i8*, i32 } %2, 1
  br label %catch

catch:
  %5 = call i8* @__cxa_begin_catch(i8* %3) #4
  %6 = bitcast i32* %0 to i8*
  call void @free(i8* %6)
  call void @__cxa_end_catch()
  br label %fun.ret

fun.ret:
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64)

; Function Attrs: nounwind
declare void @free(i8* nocapture)

declare void @boo() 

; This is an artifical example, readnone functions by definition cannot unwind
; exceptions by calling the C++ exception throwing methods
; This function should only be used to test malloc_capture.
declare void @boo_readnone() readnone

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare i32 @llvm.eh.typeid.for(i8*)

declare void @f() uwtable

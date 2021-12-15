; REQUIRES: system-darwin
; RUN: lli -jit-kind=orc-lazy %s
;
; Basic correctness testing for eh-frame processing and registration.

source_filename = "minimal-throw-catch.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@_ZTIi = external constant i8*

declare i8* @__cxa_allocate_exception(i64)
declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()

define void @explode() {
entry:
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %0 = bitcast i8* %exception to i32*
  store i32 42, i32* %0, align 16
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable
}

define i32 @main(i32 %argc, i8** %argv) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @explode()
          to label %return unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %1 = extractvalue { i8*, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:
  %3 = extractvalue { i8*, i32 } %0, 0
  %4 = tail call i8* @__cxa_begin_catch(i8* %3)
  %5 = bitcast i8* %4 to i32*
  %6 = load i32, i32* %5, align 4
  %cmp = icmp ne i32 %6, 42
  %cond = zext i1 %cmp to i32
  tail call void @__cxa_end_catch()
  br label %return

return:
  %retval.0 = phi i32 [ %cond, %catch ], [ 2, %entry ]
  ret i32 %retval.0

eh.resume:
  resume { i8*, i32 } %0
}

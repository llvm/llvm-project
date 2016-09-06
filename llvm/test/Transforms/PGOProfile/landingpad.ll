; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/landingpad.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@val = global i32 0, align 4
@_ZTIi = external constant i8*
; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_bar = private constant [3 x i8] c"bar"
; GEN: @__profn_foo = private constant [3 x i8] c"foo"

define i32 @bar(i32 %i) {
entry:
; GEN: entry:
; GEN-NOT: call void @llvm.instrprof.increment
  %rem = srem i32 %i, 3
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end
; USE: br i1 %tobool, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_BAR_ENTRY:[0-9]+]]

if.then:
; GEN: if.then:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_bar, i32 0, i32 0), i64 24868915205, i32 2, i32 1)
  %exception = call i8* @__cxa_allocate_exception(i64 4)
  %tmp = bitcast i8* %exception to i32*
  store i32 %i, i32* %tmp, align 16
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable

if.end:
; GEN: if.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_bar, i32 0, i32 0), i64 24868915205, i32 2, i32 0)
  ret i32 0
}

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

define i32 @foo(i32 %i) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; GEN: entry:
; GEN-NOT: call void @llvm.instrprof.increment
  %rem = srem i32 %i, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end
; USE: br i1 %tobool, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_FOO_ENTRY:[0-9]+]]

if.then:
; GEN: if.then:
; GEN-NOT: call void @llvm.instrprof.increment
  %mul = mul nsw i32 %i, 7
  %call = invoke i32 @bar(i32 %mul)
          to label %invoke.cont unwind label %lpad

invoke.cont:
; GEN: invoke.cont:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 59130013419, i32 4, i32 1)
  br label %if.end

lpad:
; GEN: lpad:
; GEN-NOT: call void @llvm.instrprof.increment
  %tmp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp1 = extractvalue { i8*, i32 } %tmp, 0
  %tmp2 = extractvalue { i8*, i32 } %tmp, 1
  br label %catch.dispatch

catch.dispatch:
; GEN: catch.dispatch:
; GEN-NOT: call void @llvm.instrprof.increment
  %tmp3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %tmp2, %tmp3
  br i1 %matches, label %catch, label %eh.resume
; USE: br i1 %matches, label %catch, label %eh.resume
; USE-SAME: !prof ![[BW_CATCH_DISPATCH:[0-9]+]]

catch:
; GEN: catch:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 59130013419, i32 4, i32 2)
  %tmp4 = call i8* @__cxa_begin_catch(i8* %tmp1)
  %tmp5 = bitcast i8* %tmp4 to i32*
  %tmp6 = load i32, i32* %tmp5, align 4
  %tmp7 = load i32, i32* @val, align 4
  %sub = sub nsw i32 %tmp7, %tmp6
  store i32 %sub, i32* @val, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
; GEN: try.cont:
; GEN-NOT: call void @llvm.instrprof.increment
  ret i32 -1

if.end:
; GEN: if.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 59130013419, i32 4, i32 0)
  %tmp8 = load i32, i32* @val, align 4
  %add = add nsw i32 %tmp8, %i
  store i32 %add, i32* @val, align 4
  br label %try.cont

eh.resume:
; GEN: eh.resume:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 59130013419, i32 4, i32 3)
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %tmp1, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %tmp2, 1
  resume { i8*, i32 } %lpad.val3
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

; USE: ![[BW_BAR_ENTRY]] = !{!"branch_weights", i32 2, i32 1}
; USE: ![[BW_FOO_ENTRY]] = !{!"branch_weights", i32 3, i32 2}
; USE: ![[BW_CATCH_DISPATCH]] = !{!"branch_weights", i32 2, i32 0}

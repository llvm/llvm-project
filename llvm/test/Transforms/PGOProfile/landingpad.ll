; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=false -S | FileCheck %s --check-prefixes=GEN,NOTENTRY
; RUN: llvm-profdata merge %S/Inputs/landingpad.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
;
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=true -S | FileCheck %s --check-prefixes=GEN,ENTRY
; RUN: llvm-profdata merge %S/Inputs/landingpad_entry.proftext -o %t2.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t2.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@val = global i32 0, align 4
@_ZTIi = external constant ptr
; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_bar = private constant [3 x i8] c"bar"
; GEN: @__profn_foo = private constant [3 x i8] c"foo"

define i32 @bar(i32 %i) {
entry:
; GEN: entry:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_bar, i64 {{[0-9]+}}, i32 2, i32 0)
  %rem = srem i32 %i, 3
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end
; USE: br i1 %tobool, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_BAR_ENTRY:[0-9]+]]

if.then:
; GEN: if.then:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_bar, i64 {{[0-9]+}}, i32 2, i32 1)
  %exception = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %i, ptr %exception, align 16
  call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
  unreachable

if.end:
; GEN: if.end:
; GEN-NOT: call void @llvm.instrprof.increment
; GEN: ret i32
  ret i32 0
}

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_throw(ptr, ptr, ptr)

define i32 @foo(i32 %i) personality ptr @__gxx_personality_v0 {
entry:
; GEN: entry:
; NOTENTRY-NOT: call void @llvm.instrprof.increment
; ENTRY: call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 0)
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
; GEN: call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 1)
  br label %if.end

lpad:
; GEN: lpad:
; GEN-NOT: call void @llvm.instrprof.increment
  %tmp = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %tmp1 = extractvalue { ptr, i32 } %tmp, 0
  %tmp2 = extractvalue { ptr, i32 } %tmp, 1
  br label %catch.dispatch

catch.dispatch:
; GEN: catch.dispatch:
; GEN-NOT: call void @llvm.instrprof.increment
  %tmp3 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %tmp2, %tmp3
  br i1 %matches, label %catch, label %eh.resume
; USE: br i1 %matches, label %catch, label %eh.resume
; USE-SAME: !prof ![[BW_CATCH_DISPATCH:[0-9]+]]

catch:
; GEN: catch:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 2)
  %tmp4 = call ptr @__cxa_begin_catch(ptr %tmp1)
  %tmp6 = load i32, ptr %tmp4, align 4
  %tmp7 = load i32, ptr @val, align 4
  %sub = sub nsw i32 %tmp7, %tmp6
  store i32 %sub, ptr @val, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
; GEN: try.cont:
; GEN-NOT: call void @llvm.instrprof.increment
  ret i32 -1

if.end:
; GEN: if.end:
; NOTENTRY: call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 0)
; ENTRY-NOT: call void @llvm.instrprof.increment
  %tmp8 = load i32, ptr @val, align 4
  %add = add nsw i32 %tmp8, %i
  store i32 %add, ptr @val, align 4
  br label %try.cont

eh.resume:
; GEN: eh.resume:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 3)
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %tmp1, 0
  %lpad.val3 = insertvalue { ptr, i32 } %lpad.val, i32 %tmp2, 1
  resume { ptr, i32 } %lpad.val3
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

; USE: ![[BW_BAR_ENTRY]] = !{!"branch_weights", i32 2, i32 1}
; USE: ![[BW_FOO_ENTRY]] = !{!"branch_weights", i32 3, i32 2}
; USE: ![[BW_CATCH_DISPATCH]] = !{!"branch_weights", i32 2, i32 0}

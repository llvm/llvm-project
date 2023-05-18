; RUN: opt < %s -skip-partial-inlining-cost-analysis -passes=partial-inliner -S  | FileCheck %s

declare void @bar()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define internal void @callee(i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:
  invoke void @bar()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  br label %try.cont

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %catch

catch:
  %3 = call ptr @__cxa_begin_catch(ptr %1)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  br label %if.end

if.end:
  ret void
}

define internal void @caller(i1 %cond) {
; CHECK-LABEL: define {{.*}} @caller
entry:
; CHECK: entry:
; CHECK-NEXT: br i1
; CHECK: codeRepl.i:
; CHECK-NEXT: call void @callee.1.{{.*}}()
  call void @callee(i1 %cond)
  ret void
}

; CHECK-LABEL: define {{.*}} @callee.1.{{.*}}() personality ptr @__gxx_personality_v0
; CHECK: invoke void @bar()
; CHECK: landingpad

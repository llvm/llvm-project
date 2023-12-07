; RUN: opt < %s -skip-partial-inlining-cost-analysis -passes=partial-inliner -S  | FileCheck %s

declare dso_local void @bar()
declare dso_local i32 @__CxxFrameHandler3(...)

define internal void @callee(i1 %cond) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:
  invoke void @bar()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:
  br label %try.cont

ehcleanup:
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:
  %1 = catchswitch within none [label %catch] unwind to caller

catch:
  %2 = catchpad within %1 [ptr null, i32 64, ptr null]
  catchret from %2 to label %catchret.dest

catchret.dest:
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

; CHECK-LABEL: define {{.*}} @callee.1.{{.*}}() personality ptr @__CxxFrameHandler3
; CHECK: invoke void @bar()
; CHECK: cleanuppad
; CHECK-NEXT: cleanupret
; CHECK: catchswitch
; CHECK: catchpad
; CHECK-NEXT: catchret

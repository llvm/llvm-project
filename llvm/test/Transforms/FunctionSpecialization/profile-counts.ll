; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -S < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@A = external dso_local constant i32, align 4
@B = external dso_local constant i32, align 4

; CHECK: define dso_local i32 @bar(i32 %x, i32 %y, ptr %z) !prof ![[BAR_PROF:[0-9]]] {
define dso_local i32 @bar(i32 %x, i32 %y, ptr %z) !prof !0 {
entry:
  %tobool = icmp ne i32 %x, 0
; CHECK: br i1 %tobool, label %if.then, label %if.else, !prof ![[BRANCH_PROF:[0-9]]]
  br i1 %tobool, label %if.then, label %if.else, !prof !1

; CHECK: if.then:
; CHECK: call i32 @foo.specialized.1(i32 %x, ptr @A)
if.then:
  %call = call i32 @foo(i32 %x, ptr @A)
  br label %return

; CHECK: if.else:
; CHECK: call i32 @foo.specialized.2(i32 %y, ptr @B)
if.else:
  %call1 = call i32 @foo(i32 %y, ptr @B)
  br label %return

; CHECK: return:
; CHECK: %call2 = call i32 @foo(i32 %x, ptr %z)
return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  %call2 = call i32 @foo(i32 %x, ptr %z);
  %add = add i32 %retval.0, %call2
  ret i32 %add
}

; CHECK: define internal i32 @foo(i32 %x, ptr %b) !prof ![[FOO_UNSPEC_PROF:[0-9]]]
; CHECK: define internal i32 @foo.specialized.1(i32 %x, ptr %b) !prof ![[FOO_SPEC_1_PROF:[0-9]]]
; CHECK: define internal i32 @foo.specialized.2(i32 %x, ptr %b) !prof ![[FOO_SPEC_2_PROF:[0-9]]]
define internal i32 @foo(i32 %x, ptr %b) !prof !2 {
entry:
  %0 = load i32, ptr %b, align 4
  %add = add nsw i32 %x, %0
  ret i32 %add
}

; CHECK: ![[BAR_PROF]] = !{!"function_entry_count", i64 1000}
; CHECK: ![[BRANCH_PROF]] = !{!"branch_weights", i32 1, i32 3}
; CHECK: ![[FOO_UNSPEC_PROF]] =  !{!"function_entry_count", i64 234}
; CHECK: ![[FOO_SPEC_1_PROF]] = !{!"function_entry_count", i64 250}
; CHECK: ![[FOO_SPEC_2_PROF]] = !{!"function_entry_count", i64 750}
!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 1, i32 3}
!2 = !{!"function_entry_count", i64 1234}

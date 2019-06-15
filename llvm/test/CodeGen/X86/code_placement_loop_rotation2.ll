; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -precise-rotation-cost < %s | FileCheck %s -check-prefix=CHECK-PROFILE

define void @foo() {
; Test a nested loop case when profile data is not available.
;
; CHECK-LABEL: foo:
; CHECK: callq g
; CHECK: callq h
; CHECK: callq b
; CHECK: callq e
; CHECK: callq f
; CHECK: callq c
; CHECK: callq d

entry:
  br label %header

header:
  call void @b()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !2

if.then:
  br label %header2

header2:
  call void @c()
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %if.then2, label %if.else2, !prof !2

if.then2:
  call void @d()
  br label %if.end2

if.else2:
  call void @e()
  br label %if.end2

if.end2:
  call void @f()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header2, label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header, label %end

end:
  ret void
}

define void @bar() !prof !1 {
; Test a nested loop case when profile data is available.
;
; CHECK-PROFILE-LABEL: bar:
; CHECK-PROFILE: callq h
; CHECK-PROFILE: callq b
; CHECK-PROFILE: callq g
; CHECK-PROFILE: callq e
; CHECK-PROFILE: callq f
; CHECK-PROFILE: callq c
; CHECK-PROFILE: callq d

entry:
  br label %header

header:
  call void @b()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !2

if.then:
  br label %header2

header2:
  call void @c()
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %if.then2, label %if.else2, !prof !2

if.then2:
  call void @d()
  br label %if.end2

if.else2:
  call void @e()
  br label %if.end2

if.end2:
  call void @f()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header2, label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header, label %end

end:
  ret void
}

declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()
declare void @g()
declare void @h()

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 16, i32 16}

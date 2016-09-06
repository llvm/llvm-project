; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -precise-rotation-cost < %s | FileCheck %s -check-prefix=CHECK-PROFILE

define void @foo() {
; Test that not all edges in the loop chain are fall through without profile
; data.
;
; CHECK-LABEL: foo:
; CHECK: callq e
; CHECK: callq f
; CHECK: callq g
; CHECK: callq h

entry:
  br label %header

header:
  call void @e()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !2

if.then:
  call void @f()
  br label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end

end:
  ret void
}

define void @bar() !prof !1 {
; Test that all edges in the loop chain are fall through with profile data.
;
; CHECK-PROFILE-LABEL: bar:
; CHECK-PROFILE: callq g
; CHECK-PROFILE: callq h
; CHECK-PROFILE: callq e
; CHECK-PROFILE: callq f

entry:
  br label %header

header:
  call void @e()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !2

if.then:
  call void @f()
  br label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end

end:
  ret void
}

declare zeroext i1 @a()
declare void @e()
declare void @f()
declare void @g()
declare void @h()

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 16, i32 16}

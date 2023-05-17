; Test to ensure dropping of type tests can handle a phi feeding the assume.
; RUN: opt -S -passes=lowertypetests -lowertypetests-drop-type-tests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1
@_ZTV1C = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1C1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !2

; CHECK-LABEL: define i32 @test
define i32 @test(ptr %obj, i32 %a, i32 %b) {
entry:
  %tobool.not = icmp eq i32 %a, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
; CHECK-NOT: @llvm.type.test
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)
  br label %if.end

if.else:
  %0 = icmp ne i32 %b, 0
  br label %if.end

if.end:
  %1 = phi i1 [ %0, %if.else ], [ %p, %if.then ]
  call void @llvm.assume(i1 %1)
; Still have the assume, but the type test target replaced with true.
; CHECK: %1 = phi i1 [ %0, %if.else ], [ true, %if.then ]
; CHECK: call void @llvm.assume(i1 %1)

  ret i32 0
}
; CHECK-LABEL: ret i32
; CHECK-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1C1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}

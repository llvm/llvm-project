; Test to ensure dropping of type tests can handle a phi feeding the assume.
; RUN: opt -S -passes=lowertypetests -lowertypetests-drop-type-tests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1
@_ZTV1C = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.C*, i32)* @_ZN1C1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !2

; CHECK-LABEL: define i32 @test
define i32 @test(%struct.A* %obj, i32 %a, i32 %b) {
entry:
  %tobool.not = icmp eq i32 %a, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:
  %0 = bitcast %struct.A* %obj to i8***
  %vtable = load i8**, i8*** %0
  %1 = bitcast i8** %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
; CHECK-NOT: @llvm.type.test
  %fptrptr = getelementptr i8*, i8** %vtable, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.A*, i32)**
  %fptr1 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %2, align 8
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)
  br label %if.end

if.else:
  %3 = icmp ne i32 %b, 0
  br label %if.end

if.end:
  %4 = phi i1 [ %3, %if.else ], [ %p, %if.then ]
  call void @llvm.assume(i1 %4)
; Still have the assume, but the type test target replaced with true.
; CHECK: %4 = phi i1 [ %3, %if.else ], [ true, %if.then ]
; CHECK: call void @llvm.assume(i1 %4)

  ret i32 0
}
; CHECK-LABEL: ret i32
; CHECK-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

define i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1A1nEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1C1fEi(%struct.C* %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}

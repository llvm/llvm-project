; ModuleID = 'lib.cc'
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Derived = type { %Base }
%Base = type { i32 (...)** }

@_ZTV7Derived = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%Derived*)* @_ZN7DerivedD0Ev to i8*)] }, !type !0, !type !1, !vcall_visibility !2
@_ZTV4Base = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%Base*)* @_ZN4BaseD0Ev to i8*)] }, !type !0, !vcall_visibility !2

define void @_Z3fooP4Base(%Base* %b) {
entry:
  %0 = bitcast %Base* %b to void (%Base*)***
  %vtable = load void (%Base*)**, void (%Base*)*** %0
  %1 = bitcast void (%Base*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %2)
  %vfn = getelementptr inbounds void (%Base*)*, void (%Base*)** %vtable, i64 0
  %3 = load void (%Base*)*, void (%Base*)** %vfn
  tail call void %3(%Base* %b)
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)

declare void @llvm.assume(i1)

define void @_ZN7DerivedD0Ev(%Derived* %this) {
  ret void
}

define void @_ZN4BaseD0Ev(%Base* %this) {
  unreachable
}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTS7Derived"}
!2 = !{i64 1}

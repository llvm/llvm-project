; ModuleID = 'lib.cc'
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Derived = type { %Base }
%Base = type { ptr }

@_ZTV7Derived = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN7DerivedD0Ev] }, !type !0, !type !1, !vcall_visibility !2
@_ZTV4Base = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN4BaseD0Ev] }, !type !0, !vcall_visibility !2

define void @_Z3fooP4Base(ptr %b) {
entry:
  %vtable = load ptr, ptr %b
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable
  tail call void %1(ptr %b)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)

declare void @llvm.assume(i1)

define void @_ZN7DerivedD0Ev(ptr %this) {
  ret void
}

define void @_ZN4BaseD0Ev(ptr %this) {
  unreachable
}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTS7Derived"}
!2 = !{i64 1}

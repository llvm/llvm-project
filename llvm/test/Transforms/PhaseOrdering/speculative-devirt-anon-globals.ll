; RUN: opt -S -passes='default<O3>' -enable-devirtualize-speculatively -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

@0 = private constant i32 1
@1 = private constant i32 2

@vt = constant ptr @vfunc, !type !0

define ptr @f1() { ret ptr @0 }
define ptr @f2() { ret ptr @1 }

define i32 @vfunc(ptr %obj) { ret i32 0 }

; CHECK: devirtualized a call to vfunc
define i32 @call_virtual(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %1 = call i1 @llvm.public.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %1)
  %fptr = load ptr, ptr %vtable
  %2 = call i32 %fptr(ptr %obj)
  ret i32 %2
}

!0 = !{i64 0, !"typeid"}

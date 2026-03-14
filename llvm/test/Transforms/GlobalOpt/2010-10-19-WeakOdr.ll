; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; PR8389: Globals with weak_odr linkage type must not be modified

; CHECK: weak_odr local_unnamed_addr global i32 0

@SomeVar = weak_odr global i32 0

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @CTOR, ptr null } ]

define internal void @CTOR() {
  store i32 23, ptr @SomeVar
  ret void
}



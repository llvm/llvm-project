; RUN: opt < %s -globalopt -S | FileCheck %s

; PR8389: Globals with weak_odr linkage type must not be modified

; CHECK: weak_odr local_unnamed_addr global i32 0

@SomeVar = weak_odr global i32 0

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [ { i32, void ()* } { i32 65535, void ()* @CTOR } ]

define internal void @CTOR() {
  store i32 23, i32* @SomeVar
  ret void
}



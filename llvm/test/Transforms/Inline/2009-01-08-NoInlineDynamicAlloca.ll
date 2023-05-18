; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; Do not inline calls with variable-sized alloca.

@q = common global ptr null

define ptr @a(i32 %i) nounwind {
; CHECK-LABEL: define ptr @a
entry:
  %i_addr = alloca i32
  %retval = alloca ptr
  %p = alloca ptr
  %"alloca point" = bitcast i32 0 to i32
  store i32 %i, ptr %i_addr
  %0 = load i32, ptr %i_addr, align 4
  %1 = alloca i8, i32 %0
  store ptr %1, ptr %p, align 4
  %2 = load ptr, ptr %p, align 4
  store ptr %2, ptr @q, align 4
  br label %return

return:
  %retval1 = load ptr, ptr %retval
  ret ptr %retval1
}

define void @b(i32 %i) nounwind {
; CHECK-LABEL: define void @b
entry:
  %i_addr = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store i32 %i, ptr %i_addr
  %0 = load i32, ptr %i_addr, align 4
  %1 = call ptr @a(i32 %0) nounwind
; CHECK: call ptr @a
  br label %return

return:
  ret void
}

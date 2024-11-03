; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

; array [4 x i8]
; Requires protector.

; CHECK: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr

define void @foo(ptr %a) nounwind uwtable safestack {
entry:
  ; CHECK: %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr

  ; CHECK: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16

  ; CHECK: store ptr %[[USST]], ptr @__safestack_unsafe_stack_ptr

  %a.addr = alloca ptr, align 8
  %buf = alloca [4 x i8], align 1

  ; CHECK: %[[AADDR:.*]] = alloca ptr, align 8
  ; CHECK: store ptr {{.*}}, ptr %[[AADDR]], align 8
  store ptr %a, ptr %a.addr, align 8

  ; CHECK: %[[A2:.*]] = load ptr, ptr %[[AADDR]], align 8
  ; CHECK: %[[BUFPTR:.*]] = getelementptr i8, ptr %[[USP]], i32 -4
  %a2 = load ptr, ptr %a.addr, align 8

  ; CHECK: call ptr @strcpy(ptr %[[BUFPTR]], ptr %[[A2]])
  %call = call ptr @strcpy(ptr %buf, ptr %a2)

  ; CHECK: store ptr %[[USP]], ptr @__safestack_unsafe_stack_ptr
  ret void
}

; Load from an array at a fixed offset, no overflow.
define i8 @StaticArrayFixedSafe() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayFixedSafe(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, ptr %buf, i32 2
  %x = load i8, ptr %gep, align 1
  ret i8 %x
}

; Load from an array at a fixed offset with overflow.
define i8 @StaticArrayFixedUnsafe() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayFixedUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, ptr %buf, i32 5
  %x = load i8, ptr %gep, align 1
  ret i8 %x
}

; Load from an array at an unknown offset.
define i8 @StaticArrayVariableUnsafe(i32 %ofs) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayVariableUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, ptr %buf, i32 %ofs
  %x = load i8, ptr %gep, align 1
  ret i8 %x
}

; Load from an array of an unknown size.
define i8 @DynamicArrayUnsafe(i32 %sz) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @DynamicArrayUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 %sz, align 1
  %gep = getelementptr inbounds i8, ptr %buf, i32 2
  %x = load i8, ptr %gep, align 1
  ret i8 %x
}

declare ptr @strcpy(ptr, ptr)

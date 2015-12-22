; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -safe-stack-usp-storage=single-thread -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck -check-prefix=SINGLE-THREAD %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -safe-stack-usp-storage=single-thread -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck -check-prefix=SINGLE-THREAD %s

; array [4 x i8]
; Requires protector.

; CHECK: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global i8*
; SINGLE-THREAD: @__safestack_unsafe_stack_ptr = external global i8*

define void @foo(i8* %a) nounwind uwtable safestack {
entry:
  ; CHECK: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr

  ; CHECK: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16

  ; CHECK: store i8* %[[USST]], i8** @__safestack_unsafe_stack_ptr

  ; CHECK: %[[AADDR:.*]] = alloca i8*, align 8
  %a.addr = alloca i8*, align 8

  ; CHECK: %[[BUFPTR:.*]] = getelementptr i8, i8* %[[USP]], i32 -4
  ; CHECK: %[[BUFPTR2:.*]] = bitcast i8* %[[BUFPTR]] to [4 x i8]*
  %buf = alloca [4 x i8], align 1

  ; CHECK: store i8* {{.*}}, i8** %[[AADDR]], align 8
  store i8* %a, i8** %a.addr, align 8

  ; CHECK: %[[GEP:.*]] = getelementptr inbounds [4 x i8], [4 x i8]* %[[BUFPTR2]], i32 0, i32 0
  %gep = getelementptr inbounds [4 x i8], [4 x i8]* %buf, i32 0, i32 0

  ; CHECK: %[[A2:.*]] = load i8*, i8** %[[AADDR]], align 8
  %a2 = load i8*, i8** %a.addr, align 8

  ; CHECK: call i8* @strcpy(i8* %[[GEP]], i8* %[[A2]])
  %call = call i8* @strcpy(i8* %gep, i8* %a2)

  ; CHECK: store i8* %[[USP]], i8** @__safestack_unsafe_stack_ptr
  ret void
}

; Load from an array at a fixed offset, no overflow.
define i8 @StaticArrayFixedSafe() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayFixedSafe(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, i8* %buf, i32 2
  %x = load i8, i8* %gep, align 1
  ret i8 %x
}

; Load from an array at a fixed offset with overflow.
define i8 @StaticArrayFixedUnsafe() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayFixedUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, i8* %buf, i32 5
  %x = load i8, i8* %gep, align 1
  ret i8 %x
}

; Load from an array at an unknown offset.
define i8 @StaticArrayVariableUnsafe(i32 %ofs) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @StaticArrayVariableUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 4, align 1
  %gep = getelementptr inbounds i8, i8* %buf, i32 %ofs
  %x = load i8, i8* %gep, align 1
  ret i8 %x
}

; Load from an array of an unknown size.
define i8 @DynamicArrayUnsafe(i32 %sz) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define i8 @DynamicArrayUnsafe(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret i8
  %buf = alloca i8, i32 %sz, align 1
  %gep = getelementptr inbounds i8, i8* %buf, i32 2
  %x = load i8, i8* %gep, align 1
  ret i8 %x
}

declare i8* @strcpy(i8*, i8*)

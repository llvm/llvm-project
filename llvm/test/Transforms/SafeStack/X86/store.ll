; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

define void @bad_store() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @bad_store(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %1 = inttoptr i64 %0 to ptr
  store i64 zeroinitializer, ptr %1
  ret void
}

define void @good_store() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @good_store(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  store i8 zeroinitializer, ptr %a
  ret void
}

define void @overflow_gep_store() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @overflow_gep_store(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = getelementptr i8, ptr %a, i32 4
  store i8 zeroinitializer, ptr %0
  ret void
}

define void @underflow_gep_store() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @underflow_gep_store(
  ; CHECK: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = getelementptr i8, ptr %a, i32 -1
  store i8 zeroinitializer, ptr %0
  ret void
}

define void @good_gep_store() nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: @good_gep_store(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a = alloca i32, align 4
  %0 = getelementptr i8, ptr %a, i32 3
  store i8 zeroinitializer, ptr %0
  ret void
}

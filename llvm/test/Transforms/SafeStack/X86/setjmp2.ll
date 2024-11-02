; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@buf = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; setjmp/longjmp test with dynamically sized array.
; Requires protector.
; CHECK: @foo(i32 %[[ARG:.*]])
define i32 @foo(i32 %size) nounwind uwtable safestack {
entry:
  ; CHECK: %[[SP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: %[[DYNPTR:.*]] = alloca ptr
  ; CHECK-NEXT: store ptr %[[SP]], ptr %[[DYNPTR]]

  ; CHECK-NEXT: %[[ZEXT:.*]] = zext i32 %[[ARG]] to i64
  ; CHECK-NEXT: %[[MUL:.*]] = mul i64 %[[ZEXT]], 4
  ; CHECK-NEXT: %[[SP2:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: %[[PTRTOINT:.*]] = ptrtoint ptr %[[SP2]] to i64
  ; CHECK-NEXT: %[[SUB:.*]] = sub i64 %[[PTRTOINT]], %[[MUL]]
  ; CHECK-NEXT: %[[AND:.*]] = and i64 %[[SUB]], -16
  ; CHECK-NEXT: %[[INTTOPTR:.*]] = inttoptr i64 %[[AND]] to ptr
  ; CHECK-NEXT: store ptr %[[INTTOPTR]], ptr @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: store ptr %[[INTTOPTR]], ptr %unsafe_stack_dynamic_ptr
  %a = alloca i32, i32 %size

  ; CHECK: setjmp
  ; CHECK-NEXT: %[[LOAD:.*]] = load ptr, ptr %[[DYNPTR]]
  ; CHECK-NEXT: store ptr %[[LOAD]], ptr @__safestack_unsafe_stack_ptr
  %call = call i32 @_setjmp(ptr @buf) returns_twice

  ; CHECK: call void @funcall(ptr %[[INTTOPTR]])
  call void @funcall(ptr %a)
  ; CHECK-NEXT: store ptr %[[SP:.*]], ptr @__safestack_unsafe_stack_ptr
  ret i32 0
}

declare i32 @_setjmp(ptr)
declare void @funcall(ptr)

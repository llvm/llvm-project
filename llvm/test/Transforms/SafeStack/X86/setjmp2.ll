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
  ; CHECK: %[[SP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: %[[DYNPTR:.*]] = alloca i8*
  ; CHECK-NEXT: store i8* %[[SP]], i8** %[[DYNPTR]]

  ; CHECK-NEXT: %[[ZEXT:.*]] = zext i32 %[[ARG]] to i64
  ; CHECK-NEXT: %[[MUL:.*]] = mul i64 %[[ZEXT]], 4
  ; CHECK-NEXT: %[[SP2:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: %[[PTRTOINT:.*]] = ptrtoint i8* %[[SP2]] to i64
  ; CHECK-NEXT: %[[SUB:.*]] = sub i64 %[[PTRTOINT]], %[[MUL]]
  ; CHECK-NEXT: %[[AND:.*]] = and i64 %[[SUB]], -16
  ; CHECK-NEXT: %[[INTTOPTR:.*]] = inttoptr i64 %[[AND]] to i8*
  ; CHECK-NEXT: store i8* %[[INTTOPTR]], i8** @__safestack_unsafe_stack_ptr
  ; CHECK-NEXT: store i8* %[[INTTOPTR]], i8** %unsafe_stack_dynamic_ptr
  ; CHECK-NEXT: %[[ALLOCA:.*]] = bitcast i8* %[[INTTOPTR]] to i32*
  %a = alloca i32, i32 %size

  ; CHECK: setjmp
  ; CHECK-NEXT: %[[LOAD:.*]] = load i8*, i8** %[[DYNPTR]]
  ; CHECK-NEXT: store i8* %[[LOAD]], i8** @__safestack_unsafe_stack_ptr
  %call = call i32 @_setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @buf, i32 0, i32 0)) returns_twice

  ; CHECK: call void @funcall(i32* %[[ALLOCA]])
  call void @funcall(i32* %a)
  ; CHECK-NEXT: store i8* %[[SP:.*]], i8** @__safestack_unsafe_stack_ptr
  ret i32 0
}

declare i32 @_setjmp(%struct.__jmp_buf_tag*)
declare void @funcall(i32*)

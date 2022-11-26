; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@buf = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; setjmp/longjmp test.
; Requires protector.
define i32 @foo() nounwind uwtable safestack {
entry:
  ; CHECK: %[[SP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
  ; CHECK: %[[STATICTOP:.*]] = getelementptr i8, ptr %[[SP]], i32 -16
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, ptr %retval
  store i32 42, ptr %x, align 4
  %call = call i32 @_setjmp(ptr @buf) returns_twice
  ; CHECK: setjmp
  ; CHECK-NEXT: store ptr %[[STATICTOP]], ptr @__safestack_unsafe_stack_ptr
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.else, label %if.then
if.then:                                          ; preds = %entry
  call void @funcall(ptr %x)
  br label %if.end
if.else:                                          ; preds = %entry
  call i32 (...) @dummy()
  br label %if.end
if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

declare i32 @_setjmp(ptr)
declare void @funcall(ptr)
declare i32 @dummy(...)

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
  ; CHECK: %[[SP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
  ; CHECK: %[[STATICTOP:.*]] = getelementptr i8, i8* %[[SP]], i32 -16
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 42, i32* %x, align 4
  %call = call i32 @_setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @buf, i32 0, i32 0)) returns_twice
  ; CHECK: setjmp
  ; CHECK-NEXT: store i8* %[[STATICTOP]], i8** @__safestack_unsafe_stack_ptr
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.else, label %if.then
if.then:                                          ; preds = %entry
  call void @funcall(i32* %x)
  br label %if.end
if.else:                                          ; preds = %entry
  call i32 (...) @dummy()
  br label %if.end
if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

declare i32 @_setjmp(%struct.__jmp_buf_tag*)
declare void @funcall(i32*)
declare i32 @dummy(...)

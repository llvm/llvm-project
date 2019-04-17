; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a local cast to a ptr of a different type
;   (e.g., int a; ... ; float *b = &a;)
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %a = alloca i32, align 4
  %b = alloca float*, align 8
  store i32 0, i32* %a, align 4
  %0 = bitcast i32* %a to float*
  store float* %0, float** %b, align 8
  %1 = load float*, float** %b, align 8
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), float* %1)
  ret void
}

declare i32 @printf(i8*, ...)

; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.vec = type { <4 x i32> }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a vector nested in a struct
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %c = alloca %struct.vec, align 16
  %y = getelementptr inbounds %struct.vec, %struct.vec* %c, i64 0, i32 0
  %add.ptr = getelementptr inbounds <4 x i32>, <4 x i32>* %y, i64 -12
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), <4 x i32>* %add.ptr) nounwind
  ret void
}

declare i32 @printf(i8*, ...)

; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.pair = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of struct element, GEP followed by callinst.
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair, %struct.pair* %c, i64 0, i32 1
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32* %y) nounwind
  ret void
}

declare i32 @printf(i8*, ...)

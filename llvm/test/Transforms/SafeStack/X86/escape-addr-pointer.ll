; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; Addr-of a pointer
;  safestack attribute
; Requires protector.
define void @foo() nounwind uwtable safestack {
entry:
  ; CHECK: __safestack_unsafe_stack_ptr
  %a = alloca i32*, align 8
  %b = alloca i32**, align 8
  %call = call i32* @getp()
  store i32* %call, i32** %a, align 8
  store i32** %a, i32*** %b, align 8
  %0 = load i32**, i32*** %b, align 8
  call void @funcall2(i32** %0)
  ret void
}

declare void @funcall2(i32**)
declare i32* @getp()

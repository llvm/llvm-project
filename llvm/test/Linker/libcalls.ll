; RUN: llvm-link %s %S/Inputs/strlen.ll -S -o - 2>%t.a.err | FileCheck %s --check-prefix=CHECK1
; RUN: llvm-link %S/Inputs/strlen.ll %s -S -o - 2>%t.a.err | FileCheck %s --check-prefix=CHECK2

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [7 x i8] c"string\00", align 1
@str = dso_local global ptr @.str, align 8

define void @foo() #0 {
  ret void
}

declare i64 @strlen(ptr)

define void @bar() #0 {
  ret void
}

define i64 @baz() #0 {
entry:
  %0 = load ptr, ptr @str, align 8
  %call = call i64 @strlen(ptr noundef %0)
  ret i64 %call
}

attributes #0 = { noinline }

; CHECK1: define void @foo() #[[ATTR0:[0-9]+]]
; CHECK1: define void @bar() #[[ATTR0:[0-9]+]]
; CHECK1: define i64 @baz() #[[ATTR0:[0-9]+]]
; CHECK1: define i64 @strlen(ptr [[S:%.*]]) #[[ATTR0]]

; CHECK2: define i64 @strlen(ptr [[S:%.*]]) #[[ATTR0:[0-9]+]]
; CHECK2: define void @foo() #[[ATTR0:[0-9]+]]
; CHECK2: define void @bar() #[[ATTR0:[0-9]+]]
; CHECK2: define i64 @baz() #[[ATTR0]]

; CHECK1: attributes #[[ATTR0]] = { nobuiltin noinline "no-builtins" }
; CHECK2: attributes #[[ATTR0]] = { nobuiltin noinline "no-builtins" }

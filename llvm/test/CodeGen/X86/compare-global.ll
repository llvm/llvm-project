; RUN: llc -o - %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

@foo = external global i8

define void @f(ptr %c) {
entry:
  ; CHECK: cmpl $_foo, 4(%esp)
  %cmp = icmp eq ptr %c, @foo
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @g()
  br label %if.end

if.end:
  ret void
}

declare void @g()

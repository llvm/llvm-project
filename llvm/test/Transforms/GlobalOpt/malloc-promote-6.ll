; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; CHECK-NOT: @global

@global = internal global i8* null
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @zot, i8* null }]

declare i8* @_Znwm(i64)

define internal void @widget() {
  %tmp = tail call i8* @_Znwm(i64 0)
  %tmp2 = getelementptr inbounds i8, i8* %tmp, i64 0
  store i8* %tmp, i8** @global, align 8
  call void @baz(void ()* @spam)
  ret void
}

define internal void @spam() {
  %tmp = load i8*, i8** @global, align 8
  %tmp2 = getelementptr inbounds i8, i8* %tmp, i64 0
  ret void
}

define internal void @zot() {
  call void @baz(void ()* @widget)
  ret void
}

declare void @baz(void ()*)


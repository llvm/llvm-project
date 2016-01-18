; RUN: opt %s -mergefunc -globalopt -S -o - | FileCheck %s

; Make sure we don't crash on this example. This test is supposed to test that
; MergeFunctions clears its GlobalNumbers value map. If this map still contains
; entries when running globalopt and the MergeFunctions instance is still alive
; the optimization of @G would cause an assert because globalopt would do an
; RAUW on @G which still exists as an entry in the GlobalNumbers ValueMap which
; causes an assert in the ValueHandle call back because we are RAUWing with a
; different type (AllocaInst) than its key type (GlobalValue).

@G = internal global i8** null
@G2 = internal global i8** null

define i32 @main(i32 %argc, i8** %argv) {
; CHECK: alloca
  store i8** %argv, i8*** @G
  ret i32 0
}

define internal i8** @dead1(i64 %p) {
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  %tmp = load i8**, i8*** @G
  ret i8** %tmp
}

define internal i8** @dead2(i64 %p) {
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  %tmp = load i8**, i8*** @G2
  ret i8** %tmp
}

define void @left(i64 %p) {
entry-block:
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  ret void
}

define void @right(i64 %p) {
entry-block:
  call void @left(i64 %p)
  call void @left(i64 %p)
  call void @left(i64 %p)
  call void @left(i64 %p)
  ret void
}

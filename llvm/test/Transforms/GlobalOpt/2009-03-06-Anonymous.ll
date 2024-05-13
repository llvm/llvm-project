; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@0 = global i32 0
; CHECK-DAG: @0 = internal global i32 0

@1 = private global i32 0
; CHECK-DAG: @1 = private global i32 0

define ptr @2() {
	ret ptr @0
}
; CHECK-DAG: define internal fastcc ptr @2()

define ptr @f() {
entry:
	call ptr @2()
	ret ptr %0
}

define ptr @g() {
entry:
	ret ptr @1
}

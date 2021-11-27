; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@0 = global i32 0
; CHECK-DAG: @0 = internal global i32 0

@1 = private global i32 0
; CHECK-DAG: @1 = private global i32 0

define i32* @2() {
	ret i32* @0
}
; CHECK-DAG: define internal fastcc i32* @2()

define i32* @f() {
entry:
	call i32* @2()
	ret i32* %0
}

define i32* @g() {
entry:
	ret i32* @1
}

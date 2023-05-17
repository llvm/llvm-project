; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn  -S | FileCheck %s

; PR15967
; BasicAA claims no alias when there is (due to a problem when the MaxLookup
; limit was reached).

target datalayout = "e"

%struct.foo = type { i32, i32 }

define i32 @main() {
  %t = alloca %struct.foo, align 4
  store i32 1, ptr %t, align 4
  %1 = getelementptr inbounds %struct.foo, ptr %t, i64 1
  %2 = getelementptr inbounds i8, ptr %1, i32 -1
  store i8 0, ptr %2
  %3 = getelementptr inbounds i8, ptr %2, i32 -1
  store i8 0, ptr %3
  %4 = getelementptr inbounds i8, ptr %3, i32 -1
  store i8 0, ptr %4
  %5 = getelementptr inbounds i8, ptr %4, i32 -1
  store i8 0, ptr %5
  %6 = getelementptr inbounds i8, ptr %5, i32 -1
  store i8 0, ptr %6
  %7 = getelementptr inbounds i8, ptr %6, i32 -1
  store i8 0, ptr %7
  %8 = getelementptr inbounds i8, ptr %7, i32 -1
  store i8 0, ptr %8
  %9 = getelementptr inbounds i8, ptr %8, i32 -1
  store i8 0, ptr %9
  %10 = load i32, ptr %t, align 4
  ret i32 %10
; CHECK: ret i32 %10
}

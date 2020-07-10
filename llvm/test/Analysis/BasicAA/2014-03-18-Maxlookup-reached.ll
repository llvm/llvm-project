; RUN: opt < %s -basic-aa -gvn  -S | FileCheck %s

; PR15967
; BasicAA claims no alias when there is (due to a problem when the MaxLookup
; limit was reached).

target datalayout = "e"

%struct.foo = type { i32, i32 }

define i32 @main() {
  %t = alloca %struct.foo, align 4
  %1 = getelementptr inbounds %struct.foo, %struct.foo* %t, i32 0, i32 0
  store i32 1, i32* %1, align 4
  %2 = getelementptr inbounds %struct.foo, %struct.foo* %t, i64 1
  %3 = bitcast %struct.foo* %2 to i8*
  %4 = getelementptr inbounds i8, i8* %3, i32 -1
  store i8 0, i8* %4
  %5 = getelementptr inbounds i8, i8* %4, i32 -1
  store i8 0, i8* %5
  %6 = getelementptr inbounds i8, i8* %5, i32 -1
  store i8 0, i8* %6
  %7 = getelementptr inbounds i8, i8* %6, i32 -1
  store i8 0, i8* %7
  %8 = getelementptr inbounds i8, i8* %7, i32 -1
  store i8 0, i8* %8
  %9 = getelementptr inbounds i8, i8* %8, i32 -1
  store i8 0, i8* %9
  %10 = getelementptr inbounds i8, i8* %9, i32 -1
  store i8 0, i8* %10
  %11 = getelementptr inbounds i8, i8* %10, i32 -1
  store i8 0, i8* %11
  %12 = load i32, i32* %1, align 4
  ret i32 %12
; CHECK: ret i32 %12
}

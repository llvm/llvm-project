; RUN: llc -mtriple=arm64-apple-ios7.0 -verify-machineinstrs -o - %s | FileCheck %s

%struct = type { i32, i128, i8 }

@var = global %struct zeroinitializer

define i64 @check_size() {
; CHECK-LABEL: check_size:
  %starti = ptrtoint ptr @var to i64

  %endp = getelementptr %struct, ptr @var, i64 1
  %endi = ptrtoint ptr %endp to i64

  %diff = sub i64 %endi, %starti
  ret i64 %diff
; CHECK: mov w0, #48
}

define i64 @check_field() {
; CHECK-LABEL: check_field:
  %starti = ptrtoint ptr @var to i64

  %endp = getelementptr %struct, ptr @var, i64 0, i32 1
  %endi = ptrtoint ptr %endp to i64

  %diff = sub i64 %endi, %starti
  ret i64 %diff
; CHECK: mov w0, #16
}

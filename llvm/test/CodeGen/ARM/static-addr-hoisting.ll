; RUN: llc -mtriple=thumbv7-apple-ios %s -o -  | FileCheck %s

define void @multiple_store() {
; CHECK-LABEL: multiple_store:
; CHECK: movw r[[BASE1:[0-9]+]], #16960
; CHECK: movs [[VAL:r[0-9]+]], #42
; CHECK: movt r[[BASE1]], #15

; CHECK: str [[VAL]], [r[[BASE1]]]
; CHECK: str [[VAL]], [r[[BASE1]], #24]
; CHECK: str.w [[VAL]], [r[[BASE1]], #42]

; CHECK: movw r[[BASE2:[0-9]+]], #20394
; CHECK: movt r[[BASE2]], #18

; CHECK: str [[VAL]], [r[[BASE2]]]
  store i32 42, i32* inttoptr(i32 1000000 to i32*)
  store i32 42, i32* inttoptr(i32 1000024 to i32*)
  store i32 42, i32* inttoptr(i32 1000042 to i32*)
  store i32 42, i32* inttoptr(i32 1200042 to i32*)
  ret void
}

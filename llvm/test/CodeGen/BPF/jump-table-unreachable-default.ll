; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s

declare void @use(i64)

; The default is unreachable: an enum with 16 variants. The verifier doesn't know
; that, the range check has to be there. What is out of range goes where a case
; goes.
define void @enum(ptr %p) {
; CHECK-LABEL: enum:
; CHECK:       w1 = *(u8 *)(r1 + 0)
; CHECK:       if w1 > 1{{[45]}} goto
; CHECK:       gotox
entry:
  %v = load i8, ptr %p
  switch i8 %v, label %default [
    i8 0, label %c0
    i8 1, label %c1
    i8 2, label %c2
    i8 3, label %c3
    i8 4, label %c4
    i8 5, label %c5
    i8 6, label %c6
    i8 7, label %c7
    i8 8, label %c8
    i8 9, label %c9
    i8 10, label %c10
    i8 11, label %c11
    i8 12, label %c12
    i8 13, label %c13
    i8 14, label %c14
    i8 15, label %c15
  ]
c0:
  call void @use(i64 1)
  br label %out
c1:
  call void @use(i64 8)
  br label %out
c2:
  call void @use(i64 15)
  br label %out
c3:
  call void @use(i64 22)
  br label %out
c4:
  call void @use(i64 29)
  br label %out
c5:
  call void @use(i64 36)
  br label %out
c6:
  call void @use(i64 43)
  br label %out
c7:
  call void @use(i64 50)
  br label %out
c8:
  call void @use(i64 57)
  br label %out
c9:
  call void @use(i64 64)
  br label %out
c10:
  call void @use(i64 71)
  br label %out
c11:
  call void @use(i64 78)
  br label %out
c12:
  call void @use(i64 85)
  br label %out
c13:
  call void @use(i64 92)
  br label %out
c14:
  call void @use(i64 99)
  br label %out
c15:
  call void @use(i64 106)
  br label %out
default:
  unreachable
out:
  ret void
}

; No compare for the third case and nothing for a fourth one.
define i64 @three(i8 %v) {
; CHECK-LABEL: three:
; CHECK:       if w{{[0-9]}} == {{[0-9]+}} goto
; CHECK:       if w{{[0-9]}} {{[!=]}}= {{[0-9]+}} goto
; CHECK-NOT:   if
; CHECK:       .Lfunc_end1:
entry:
  switch i8 %v, label %default [
    i8 0, label %a
    i8 1, label %b
    i8 -1, label %c
  ]
a:
  call void @use(i64 1)
  ret i64 10
b:
  call void @use(i64 2)
  ret i64 20
c:
  call void @use(i64 3)
  ret i64 30
default:
  unreachable
}

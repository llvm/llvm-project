; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s

declare void @use(i64)

; The index is masked and the table has 32 entries: the load is within the table
; whatever the verifier makes of the range check.
define void @masked(i8 %v) {
; CHECK-LABEL: masked:
; CHECK:       r[[I:[0-9]]] &= 31
; CHECK-NEXT:  r[[I]] <<= 3
; CHECK:       gotox
; CHECK:       .size BPF.JT.0.{{[0-9]}}, 256
entry:
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
    i8 16, label %c16
    i8 17, label %c17
    i8 18, label %c18
    i8 19, label %c19
    i8 20, label %c20
    i8 21, label %c21
    i8 22, label %c22
    i8 23, label %c23
    i8 24, label %c24
  ]
c0:
  call void @use(i64 1)
  br label %default
c1:
  call void @use(i64 8)
  br label %default
c2:
  call void @use(i64 15)
  br label %default
c3:
  call void @use(i64 22)
  br label %default
c4:
  call void @use(i64 29)
  br label %default
c5:
  call void @use(i64 36)
  br label %default
c6:
  call void @use(i64 43)
  br label %default
c7:
  call void @use(i64 50)
  br label %default
c8:
  call void @use(i64 57)
  br label %default
c9:
  call void @use(i64 64)
  br label %default
c10:
  call void @use(i64 71)
  br label %default
c11:
  call void @use(i64 78)
  br label %default
c12:
  call void @use(i64 85)
  br label %default
c13:
  call void @use(i64 92)
  br label %default
c14:
  call void @use(i64 99)
  br label %default
c15:
  call void @use(i64 106)
  br label %default
c16:
  call void @use(i64 113)
  br label %default
c17:
  call void @use(i64 120)
  br label %default
c18:
  call void @use(i64 127)
  br label %default
c19:
  call void @use(i64 134)
  br label %default
c20:
  call void @use(i64 141)
  br label %default
c21:
  call void @use(i64 148)
  br label %default
c22:
  call void @use(i64 155)
  br label %default
c23:
  call void @use(i64 162)
  br label %default
c24:
  call void @use(i64 169)
  br label %default
default:
  ret void
}

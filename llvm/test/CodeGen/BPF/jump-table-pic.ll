; RUN: llc -mtriple=bpfel -mcpu=v4 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -relocation-model=static < %s | FileCheck %s

declare void @use(i64)

; Position independent code: the entries of the table are addresses of blocks all
; the same. The address of the table is not added to what is loaded from it.
define void @pic(i64 %v) {
; CHECK-LABEL: pic:
; CHECK:       if r1 > 15 goto
; CHECK:       r1 <<= 3
; CHECK-NEXT:  r2 = BPF.JT.0.0 ll
; CHECK-NEXT:  r2 += r1
; CHECK-NEXT:  r1 = *(u64 *)(r2 + 0)
; CHECK-NEXT:  gotox r1
entry:
  switch i64 %v, label %default [
    i64 0, label %c0
    i64 1, label %c1
    i64 2, label %c2
    i64 3, label %c3
    i64 4, label %c4
    i64 5, label %c5
    i64 6, label %c6
    i64 7, label %c7
    i64 8, label %c8
    i64 9, label %c9
    i64 10, label %c10
    i64 11, label %c11
    i64 12, label %c12
    i64 13, label %c13
    i64 14, label %c14
    i64 15, label %c15
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
default:
  ret void
}

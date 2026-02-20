; RUN: llc -mtriple s390x-zos < %s | FileCheck %s

define void @jumptable(i32 signext %in, ptr %out) {
; CHECK-LABEL: jumptable DS 0H
; CHECK:   larl 3,L#JTI0_0
; CHECK: L#func_end1 DS 0H
; CHECK: L#JTI0_0 DS 0H
; CHECK:   DC AD(L#BB0_2)
; CHECK:   DC AD(L#BB0_5)
; CHECK:   DC AD(L#BB0_3)
; CHECK:   DC AD(L#BB0_4)

entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
exit:
  ret void
}

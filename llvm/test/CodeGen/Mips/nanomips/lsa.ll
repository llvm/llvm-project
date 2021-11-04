; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define i8* @lsa0(i8* %num, i32 %n) {
; CHECK-NOT: lsa $a0, $a1, $a0, 0
; CHECK: addu $a0, $a0, $a1
  %arrayidx = getelementptr inbounds i8, i8* %num, i32 %n
  ret i8* %arrayidx
}

define i16* @lsa1(i16* %num, i32 %n) {
; CHECK: lsa $a0, $a1, $a0, 1
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 %n
  ret i16* %arrayidx
}

define i32* @lsa2(i32* %num, i32 %n) {
; CHECK: lsa $a0, $a1, $a0, 2
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 %n
  ret i32* %arrayidx
}

define i64* @lsa3(i64* %num, i32 %n) {
; CHECK: lsa $a0, $a1, $a0, 3
  %arrayidx = getelementptr inbounds i64, i64* %num, i32 %n
  ret i64* %arrayidx
}

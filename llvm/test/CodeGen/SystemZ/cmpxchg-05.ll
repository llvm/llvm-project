; Test proper extension of 8-bit/16-bit cmpxchg.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; CHECK-LABEL: f1
; CHECK: crjlh
; CHECK-NOT: llcr
; CHECK-NOT: cr
; CHECK: llgcr %r2, [[RES:%r[0-9]+]]
; CHECK-NOT: llcr
; CHECK-NOT: cr
define zeroext i8 @f1(i8* nocapture, i8 zeroext, i8 zeroext) {
  %cx = cmpxchg i8* %0, i8 %1, i8 %2 seq_cst seq_cst
  %res = extractvalue { i8, i1 } %cx, 0
  ret i8 %res
}

; CHECK-LABEL: f2
; CHECK: crjlh
; CHECK-NOT: llhr
; CHECK-NOT: cr
; CHECK: llghr %r2, [[RES:%r[0-9]+]]
; CHECK-NOT: llhr
; CHECK-NOT: cr
define zeroext i16 @f2(i16* nocapture, i16 zeroext, i16 zeroext) {
  %cx = cmpxchg i16* %0, i16 %1, i16 %2 seq_cst seq_cst
  %res = extractvalue { i16, i1 } %cx, 0
  ret i16 %res
}

; CHECK-LABEL: f3
; CHECK: crjlh
; CHECK-NOT: llcr
; CHECK-NOT: cr
; CHECK: lgbr %r2, [[RES:%r[0-9]+]]
; CHECK-NOT: llcr
; CHECK-NOT: cr
define signext i8 @f3(i8* nocapture, i8 signext, i8 signext) {
  %cx = cmpxchg i8* %0, i8 %1, i8 %2 seq_cst seq_cst
  %res = extractvalue { i8, i1 } %cx, 0
  ret i8 %res
}

; CHECK-LABEL: f4
; CHECK: crjlh
; CHECK-NOT: llhr
; CHECK-NOT: cr
; CHECK: lghr %r2, [[RES:%r[0-9]+]]
; CHECK-NOT: llhr
; CHECK-NOT: cr
define signext i16 @f4(i16* nocapture, i16 signext, i16 signext) {
  %cx = cmpxchg i16* %0, i16 %1, i16 %2 seq_cst seq_cst
  %res = extractvalue { i16, i1 } %cx, 0
  ret i16 %res
}


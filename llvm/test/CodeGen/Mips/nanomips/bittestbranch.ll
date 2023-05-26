; RUN: llc -mtriple=nanomips -stop-after=finalize-isel < %s | FileCheck %s

; NOTE: Internal linkage type for all functions in this test was
;       removed. Otherwise, this test will not pass due to the
;       problems caused by registration of the NanoMips target.

; Check correct use of bittest branches
; CHECK-LABEL: name:{{.*}}f16
define i32 @f16(i32 %x) {
  %and = and i32 %x, 16
  %masked = icmp ne i32 %and, 0
; CHECK: BBNEZC_NM {{.*}}, 4, %[[ret1:[0-9a-z\.]+]]
  br i1 %masked, label %a, label %b

b:
; CHECK-LABEL: RetRA
  ret i32 0

a:
; CHECK: [[ret1]].a:
; CHECK: Li_NM 1
; CHECK-LABEL: RetRA
  ret i32 1
}

; CHECK-LABEL: name:{{.*}}f16eq
define i32 @f16eq(i32 %x) {
  %and = and i32 %x, 16
  %masked = icmp eq i32 %and, 0
  br i1 %masked, label %a, label %b
; CHECK: BBEQZC_NM {{.*}}, 4, %[[ret1:[0-9a-z\.]+]]

b:
; CHECK-LABEL: RetRA
  ret i32 0

a:
; CHECK: [[ret1]].a:
; CHECK: Li_NM 1
; CHECK-LABEL: RetRA
  ret i32 1

}


; Check that bit tests are definitely not used for non-power-of-two ANDs.
; CHECK-LABEL: name:{{.*}}f17
define i32 @f17(i32 %x) {
  %and = and i32 %x, 17
  %masked = icmp ne i32 %and, 0
; CHECK-NOT: BBNEZC_NM
  br i1 %masked, label %a, label %b

b:
; CHECK-LABEL: RetRA
  ret i32 0

a:
; CHECK-LABEL: RetRA
  ret i32 1
}

; CHECK-LABEL: name:{{.*}}f17eq
define i32 @f17eq(i32 %x) {
  %and = and i32 %x, 17  %masked = icmp eq i32 %and, 0
  br i1 %masked, label %a, label %b
; CHECK-NOT: BBEQZC_NM

b:
; CHECK-LABEL: RetRA
  ret i32 0

a:
; CHECK-LABEL: RetRA
  ret i32 1

}


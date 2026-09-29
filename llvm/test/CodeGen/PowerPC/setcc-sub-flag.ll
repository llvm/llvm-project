; RUN: llc -mtriple=powerpc64le -simplify-mir -verify-machineinstrs \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s

declare void @foo(i64)
declare void @bar(i1)

define void @f(i64 %a, i64 %b) {
  ; CHECK-LABEL: name: f
  ; CHECK: bb.0 (%ir-block.0):
  ; CHECK:   liveins: $x3, $x4
  ; CHECK:   [[COPY:%[0-9]+]]:g8rc = COPY $x4
  ; CHECK:   [[COPY1:%[0-9]+]]:g8rc = COPY $x3
  ; CHECK:   [[SUBF8_:%[0-9]+]]:g8rc = nsw SUBF8 [[COPY1]], [[COPY]]
  %c = sub nsw i64 %b, %a
  call void @foo(i64 %c)
  %d = icmp slt i64 %a, %b
  call void @bar(i1 %d)
  ret void
}

define void @g(i64 %a, i64 %b) {
  ; CHECK-LABEL: name: g
  ; CHECK: bb.0 (%ir-block.0):
  ; CHECK:   liveins: $x3, $x4
  ; CHECK:   [[COPY:%[0-9]+]]:g8rc = COPY $x4
  ; CHECK:   [[COPY1:%[0-9]+]]:g8rc = COPY $x3
  ; CHECK:   [[SUBF8_:%[0-9]+]]:g8rc = nsw SUBF8 [[COPY]], [[COPY1]]
  %c = sub nsw i64 %a, %b
  call void @foo(i64 %c)
  %d = icmp slt i64 %a, %b
  call void @bar(i1 %d)
  ret void
}

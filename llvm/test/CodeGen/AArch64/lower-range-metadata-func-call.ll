; RUN: llc < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; and can be eliminated
; CHECK-LABEL: {{^}}test_call_known_max_range:
; CHECK: bl foo
; CHECK-NOT: and
; CHECK: ret
define i32 @test_call_known_max_range() #0 {
entry:
  %id = tail call i32 @foo(), !range !0
  %and = and i32 %id, 1023
  ret i32 %and
}

; CHECK-LABEL: {{^}}test_call_known_trunc_1_bit_range:
; CHECK: bl foo
; CHECK: and w{{[0-9]+}}, w0, #0x1ff
; CHECK: ret
define i32 @test_call_known_trunc_1_bit_range() #0 {
entry:
  %id = tail call i32 @foo(), !range !0
  %and = and i32 %id, 511
  ret i32 %and
}

; CHECK-LABEL: {{^}}test_call_known_max_range_m1:
; CHECK: bl foo
; CHECK: and w{{[0-9]+}}, w0, #0xff
; CHECK: ret
define i32 @test_call_known_max_range_m1() #0 {
entry:
  %id = tail call i32 @foo(), !range !1
  %and = and i32 %id, 255
  ret i32 %and
}


declare i32 @foo()

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind readnone }

!0 = !{i32 0, i32 1024}
!1 = !{i32 0, i32 1023}

; RUN: llc -mtriple=aarch64-unknown-linux-gnu -aarch64-tbz-offset-bits=4 -aarch64-cbz-offset-bits=4 < %s | FileCheck %s

;; Check that branch relaxation accounts for the size of xray EXIT sleds
;; Note that TAIL_CALL sleds don't exist on AArch64 and don't need a test.
define void @exit(i1 zeroext %0) nounwind "function-instrument"="xray-always" {
; CHECK-LABEL: exit:
; CHECK-NEXT:     .Lfunc_begin0:
; CHECK-NEXT:     // %bb.0:
; CHECK-NEXT:       .p2align 2
; CHECK-NEXT:     .Lxray_sled_0:
; CHECK-NEXT:       b #32
; CHECK-COUNT-7:    nop
; CHECK-NOT:        nop
; CHECK:            tbnz
; CHECK-SAME:            [[FALLTHROUGH:.LBB[0-9_]+]]
; CHECK-NEXT:       b
; CHECK-SAME:         [[OUT_OF_RANGE:.LBB[0-9_]+]]
; CHECK-NEXT:     [[FALLTHROUGH]]:
; CHECK-NEXT:       bl      bar
; CHECK:            .p2align 2
; CHECK-NEXT:     .Lxray_sled_1:
; CHECK-NEXT:       b #32
; CHECK-COUNT-7:    nop
; CHECK-NOT:        nop
; CHECK-NEXT:     .Ltmp1:
; CHECK-NEXT:       ret
; CHECK-NEXT:     [[OUT_OF_RANGE]]:
; CHECK-SAME:                        // %end2
; CHECK-NEXT:       bl      baz
  br i1 %0, label %end1, label %end2

end1:
  %2 = call i32 @bar()
  ret void

end2:
  %3 = call i32 @baz()
  ret void
}

declare i32 @bar()
declare i32 @baz()
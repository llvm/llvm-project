; RUN: llc -mtriple=aarch64-unknown-linux-gnu -aarch64-tbz-offset-bits=4 -aarch64-cbz-offset-bits=3 < %s | FileCheck %s

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

;; Check that branch relaxation accounts for the size of xray EVENT sleds
define void @customevent(i1 zeroext %0, ptr nocapture noundef readonly %e1, i64 noundef %s1, ptr nocapture noundef readonly %e2, i64 noundef %s2) "function-instrument"="xray-always" {
; CHECK-LABEL: customevent:
; CHECK-NEXT:     .Lfunc_begin1:
; CHECK-NEXT:       .cfi_startproc
; CHECK-NEXT:       // %bb.0:
; CHECK-NEXT:       .p2align 2
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK:            cbnz
; CHECK-SAME:           [[FALLTHROUGH_2:.LBB[0-9_]+]]
; CHECK-NEXT:       b
; CHECK-SAME:           [[OUT_OF_RANGE_2:.LBB[0-9_]+]]
; CHECK-NEXT:     [[FALLTHROUGH_2]]:
; CHECK-SAME:                        // %end1
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK-NEXT:         Begin XRay custom event
; CHECK:            bl      __xray_CustomEvent
; CHECK:              End XRay custom event
; CHECK-NEXT:     [[OUT_OF_RANGE_2]]:
; CHECK-SAME:                        // %end2
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK-NEXT:         Begin XRay custom event
; CHECK:            bl      __xray_CustomEvent
; CHECK:              End XRay custom event
; CHECK:          .Ltmp
; CHECK-NEXT:       ret
entry:
  br i1 %0, label %end1, label %end2

end1:
  call void @llvm.xray.customevent(ptr %e1, i64 %s1)
  br label %end2

end2:
  tail call void @llvm.xray.customevent(ptr %e2, i64 %s2)
  ret void
}

;; Check that branch relaxation accounts for the size of xray TYPED_EVENT sleds
define void @typedevent(i1 zeroext %0, i64 noundef %type, ptr nocapture noundef readonly %event, i64 noundef %size) "function-instrument"="xray-always" {
; CHECK-LABEL: typedevent:
; CHECK-NEXT:     .Lfunc_begin2:
; CHECK-NEXT:       .cfi_startproc
; CHECK-NEXT:       // %bb.0:
; CHECK-NEXT:       .p2align 2
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK:            cbnz
; CHECK-SAME:           [[FALLTHROUGH_3:.LBB[0-9_]+]]
; CHECK-NEXT:       b
; CHECK-SAME:           [[OUT_OF_RANGE_3:.LBB[0-9_]+]]
; CHECK-NEXT:     [[FALLTHROUGH_3]]:
; CHECK-SAME:                        // %end1
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK-NEXT:         Begin XRay typed event
; CHECK:            bl      __xray_TypedEvent
; CHECK:              End XRay typed event
; CHECK-NEXT:     [[OUT_OF_RANGE_3]]:
; CHECK-SAME:                        // %end2
; CHECK-NEXT:     .Lxray_sled_{{[0-9]+}}:
; CHECK-NEXT:         Begin XRay typed event
; CHECK:            bl      __xray_TypedEvent
; CHECK:              End XRay typed event
; CHECK:          .Ltmp
; CHECK-NEXT:       ret
entry:
  br i1 %0, label %end1, label %end2

end1:
  call void @llvm.xray.typedevent(i64 %type, ptr %event, i64 %size)
  br label %end2
  
end2:
  tail call void @llvm.xray.typedevent(i64 %size, ptr %event, i64 %type)
  ret void
}

declare void @llvm.xray.customevent(ptr, i64)
declare void @llvm.xray.typedevent(i64, ptr, i64)
declare i32 @bar()
declare i32 @baz()
; RUN: llc < %s -O3 -mtriple=aarch64-linux-gnu -aarch64-enable-gep-opt | FileCheck %s

%struct = type { i32, i32, i32 }

define i32 @test1(%struct* %ptr, i64 %idx) {
; CHECK-LABEL: test1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #12
; CHECK-NEXT:    madd x8, x1, x8, x0
; CHECK-NEXT:    ldr w9, [x8, #4]
; CHECK-NEXT:    tbnz w9, #31, .LBB0_2
; CHECK-NEXT:  // %bb.1:
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    ret
; CHECK-NEXT:  .LBB0_2: // %then
; CHECK-NEXT:    ldr w8, [x8, #8]
; CHECK-NEXT:    add w0, w9, w8
; CHECK-NEXT:    ret
 %gep.1 = getelementptr %struct, %struct* %ptr, i64 %idx, i32 1
 %lv.1 = load i32, i32* %gep.1
 %c = icmp slt i32 %lv.1, 0
 br i1 %c, label %then, label %else

then:
 %gep.2 = getelementptr %struct, %struct* %ptr, i64 %idx, i32 2
 %lv.2 = load i32, i32* %gep.2
 %res = add i32 %lv.1, %lv.2
 ret i32 %res

else:
 ret i32 0
}

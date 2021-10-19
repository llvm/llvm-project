; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+lob,+mve.fp -disable-arm-loloops=true %s -o - | FileCheck %s --check-prefix=DISABLED
; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+lob,+mve.fp %s -o - | FileCheck %s

; DISABLED-NOT: dls lr,

; CHECK-LABEL: test_target_specific:
; CHECK:        mov.w lr, #50
; CHECK-NOT:    dls lr, lr
; CHECK-NOT:    mov lr,
; CHECK:      [[LOOP_HEADER:\.LBB[0-9_]+]]:
; CHECK:        le lr, [[LOOP_HEADER]]
; CHECK-NOT:    b .
; CHECK:      @ %exit

define i32 @test_target_specific(i32* %a, i32* %b) {
entry:
  br label %loop
loop:
  %acc = phi i32 [ 0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr i32, i32* %a, i32 %count
  %addr.b = getelementptr i32, i32* %b, i32 %count
  %load.a = load i32, i32* %addr.a
  %load.b = load i32, i32* %addr.b
  %res = call i32 @llvm.arm.smlad(i32 %load.a, i32 %load.b, i32 %acc)
  %count.next = add nuw i32 %count, 2
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %res
}

; CHECK-LABEL: test_fabs:
; CHECK:        mov.w lr, #100
; CHECK-NOT:    dls lr, lr
; CHECK-NOT:    mov lr,
; CHECK:      [[LOOP_HEADER:\.LBB[0-9_]+]]:
; CHECK-NOT:    bl
; CHECK:        le lr, [[LOOP_HEADER]]
; CHECK-NOT:    b .
; CHECK:      @ %exit

define float @test_fabs(float* %a) {
entry:
  br label %loop
loop:
  %acc = phi float [ 0.0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr float, float* %a, i32 %count
  %load.a = load float, float* %addr.a
  %abs = call float @llvm.fabs.f32(float %load.a)
  %res = fadd float %abs, %acc
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret float %res
}

declare i32 @llvm.arm.smlad(i32, i32, i32)
declare float @llvm.fabs.f32(float)

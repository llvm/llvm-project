; RUN: llc < %s -mtriple=thumbv8m.base-arm-none-eabi | FileCheck %s

define void @fn() {
entry:
; CHECK-LABEL: fn:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    ldr r6, .LCPI0_0
; CHECK-NEXT:    .pad #1600
; CHECK-NEXT:    add sp, r6
; CHECK: .LCPI0_0:
; CHECK_NEXT:    long   4294963196
  %a = alloca [400 x i32], align 4
  %arraydecay = getelementptr inbounds [400 x i32], [400 x i32]* %a, i32 0, i32 0
  call void @bar(i32* %arraydecay)
  ret void
}

define void @execute_only_fn() #0 {
entry:
; CHECK-LABEL: execute_only_fn:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    movw    r6, #63936
; CHECK-NEXT:    movt    r6, #65535
; CHECK-NEXT:    .pad #1600
; CHECK-NEXT:    add sp, r6
  %a = alloca [400 x i32], align 4
  %arraydecay = getelementptr inbounds [400 x i32], [400 x i32]* %a, i32 0, i32 0
  call void @bar(i32* %arraydecay)
  ret void
}

declare dso_local void @bar(i32*)

attributes #0 = { noinline optnone "target-features"="+armv8-m.base,+execute-only,+thumb-mode" }

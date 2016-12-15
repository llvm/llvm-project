; RUN: llc -mtriple=thumbv8m.base-eabi -arm-execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2BASE %s
; RUN: llc -mtriple=thumbv7m-eabi      -arm-execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2 %s
; RUN: llc -mtriple=thumbv8m.main-eabi -arm-execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2 %s

@var = global i32 0

define i32 @global() minsize {
; CHECK-LABEL: global:
; CHECK: movw [[GLOBDEST:r[0-9]+]], :lower16:var
; CHECK: movt [[GLOBDEST]], :upper16:var

  %val = load i32, i32* @var
  ret i32 %val
}

define i32 @jump_table(i32 %c, i32 %a, i32 %b) #0 {
; CHECK-LABEL: jump_table:
; CHECK-T2: adr.w   [[REG_JT:r[0-9]+]], .LJTI1_0
; CHECK-T2: add.w   [[REG_ENTRY:r[0-9]+]], [[REG_JT]], {{r[0-9]+}}, lsl #2
; CHECK-T2: mov     pc, [[REG_ENTRY]]

; CHECK-T2BASE: lsls    [[REG_OFFSET:r[0-9]+]], {{r[0-9]+}}, #2
; CHECK-T2BASE: adr     [[REG_JT:r[0-9]+]], .LJTI1_0
; CHECK-T2BASE: adds    [[REG_ENTRY:r[0-9]+]], [[REG_OFFSET]], [[REG_JT]]
; CHECK-T2BASE: mov     pc, [[REG_ENTRY]]

; CHECK-LABEL: .LJTI1_0:
; CHECK-NEXT: b.w
; CHECK-NEXT: b.w
; CHECK-NEXT: b.w
; CHECK-NEXT: b.w
; CHECK-NEXT: b.w
; CHECK-NEXT: b.w

entry:
  switch i32 %c, label %return [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
    i32 5, label %sw.bb6
    i32 6, label %sw.bb8
  ]

sw.bb:                                            ; preds = %entry
  %add = add nsw i32 %a, 6
  br label %return

sw.bb1:                                           ; preds = %entry
  %add2 = add nsw i32 %a, 4
  br label %return

sw.bb3:                                           ; preds = %entry
  %sub = add nsw i32 %a, -3
  br label %return

sw.bb4:                                           ; preds = %entry
  %add5 = add nsw i32 %b, 5
  br label %return

sw.bb6:                                           ; preds = %entry
  %add7 = add nsw i32 %a, 1
  br label %return

sw.bb8:                                           ; preds = %entry
  %add9 = add nsw i32 %a, 2
  br label %return

return:                                           ; preds = %entry, %sw.bb8, %sw.bb6, %sw.bb4, %sw.bb3, %sw.bb1, %sw.bb
  %retval.0 = phi i32 [ %add9, %sw.bb8 ], [ %add7, %sw.bb6 ], [ %add5, %sw.bb4 ], [ %sub, %sw.bb3 ], [ %add2, %sw.bb1 ], [ %add, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

@.str = private unnamed_addr constant [4 x i8] c"FOO\00", align 1

define hidden i8* @string_literal() {
entry:
; CHECK-LABEL: string_literal:
; CHECK-NOT: .asciz
; CHECK: .fnend
    ret i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0)
}

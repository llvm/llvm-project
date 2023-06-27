; UNSUPPORTED: expensive_checks

; RUN: llc -mtriple=thumbv8m.base-eabi -mattr=+execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2BASE %s
; RUN: llc -mtriple=thumbv8m.base-eabi -mcpu=cortex-m23 -mattr=+execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2BASE %s
; RUN: llc -mtriple=thumbv7m-eabi      -mattr=+execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2 %s
; RUN: llc -mtriple=thumbv8m.main-eabi -mattr=+execute-only %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-T2 %s
; RUN: llc -mtriple=thumbv6m-eabi -mattr=+execute-only %s -o - | FileCheck --check-prefix=CHECK-T1 %s

; CHECK-NOT: {{^ *}}.text{{$}}
; CHECK: .section .text,"axy",%progbits,unique,0

@var = global i32 0

define i32 @global() minsize {
; CHECK-LABEL: global:
; CHECK: movw [[GLOBDEST:r[0-9]+]], :lower16:var
; CHECK-NEXT: movt [[GLOBDEST]], :upper16:var
; CHECK-T1-LABEL: global:
; CHECK-T1: movs [[GLOBDEST:r[0-9]+]], :upper8_15:var
; CHECK-T1-NEXT: lsls [[GLOBDEST]], [[GLOBDEST]], #8
; CHECK-T1-NEXT: adds [[GLOBDEST]], :upper0_7:var
; CHECK-T1-NEXT: lsls [[GLOBDEST]], [[GLOBDEST]], #8
; CHECK-T1-NEXT: adds [[GLOBDEST]], :lower8_15:var
; CHECK-T1-NEXT: lsls [[GLOBDEST]], [[GLOBDEST]], #8
; CHECK-T1-NEXT: adds [[GLOBDEST]], :lower0_7:var

  %val = load i32, ptr @var
  ret i32 %val
}

define i32 @jump_table(i32 %c, i32 %a, i32 %b) #0 {
; CHECK-LABEL: jump_table:
; CHECK-T2: adr.w   [[REG_JT:r[0-9]+]], .LJTI1_0
; CHECK-T2: add.w   [[REG_ENTRY:r[0-9]+]], [[REG_JT]], {{r[0-9]+}}, lsl #2
; CHECK-T2: mov     pc, [[REG_ENTRY]]

; CHECK-T2BASE: lsls    [[REG_OFFSET:r[0-9]+]], {{r[0-9]+}}, #2
; CHECK-T2BASE: adr     [[REG_JT:r[0-9]+]], .LJTI1_0
; CHECK-T2BASE: adds    [[REG_ENTRY:r[0-9]+]], [[REG_JT]], [[REG_OFFSET]]
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

define hidden ptr @string_literal() {
entry:
; CHECK-LABEL: string_literal:
; CHECK: movw [[STRLIT:r[0-9]+]], :lower16:.L.str
; CHECK-NEXT: movt [[STRLIT]], :upper16:.L.str
; CHECK-T1-LABEL: string_literal:
; CHECK-T1: movs [[STRLIT:r[0-9]+]], :upper8_15:.L.str
; CHECK-T1-NEXT: lsls [[STRLIT]], [[STRLIT]], #8
; CHECK-T1-NEXT: adds [[STRLIT]], :upper0_7:.L.str
; CHECK-T1-NEXT: lsls [[STRLIT]], [[STRLIT]], #8
; CHECK-T1-NEXT: adds [[STRLIT]], :lower8_15:.L.str
; CHECK-T1-NEXT: lsls [[STRLIT]], [[STRLIT]], #8
; CHECK-T1-NEXT: adds [[STRLIT]], :lower0_7:.L.str

    ret ptr @.str
}

@external_global = external global i32
define i32 @test_external_global() {
entry:
; CHECK-LABEL: external_global:
; CHECK: movw [[EXTGLOB:r[0-9]+]], :lower16:external_global
; CHECK-NEXT: movt [[EXTGLOB]], :upper16:external_global
; CHECK-T1-LABEL: external_global:
; CHECK-T1: movs [[EXTGLOB:r[0-9]+]], :upper8_15:external_global
; CHECK-T1-NEXT: lsls [[EXTGLOB]], [[EXTGLOB]], #8
; CHECK-T1-NEXT: adds [[EXTGLOB]], :upper0_7:external_global
; CHECK-T1-NEXT: lsls [[EXTGLOB]], [[EXTGLOB]], #8
; CHECK-T1-NEXT: adds [[EXTGLOB]], :lower8_15:external_global
; CHECK-T1-NEXT: lsls [[EXTGLOB]], [[EXTGLOB]], #8
; CHECK-T1-NEXT: adds [[EXTGLOB]], :lower0_7:external_global

  %v = load i32, ptr @external_global
  ret i32 %v
}

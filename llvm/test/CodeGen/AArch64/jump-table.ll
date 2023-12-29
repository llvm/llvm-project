; RUN: llc -no-integrated-as -verify-machineinstrs -o - %s -aarch64-min-jump-table-entries=4 -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc -no-integrated-as -code-model=large -verify-machineinstrs -o - %s -aarch64-min-jump-table-entries=4 -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck --check-prefix=CHECK-LARGE %s
; RUN: llc -no-integrated-as -code-model=large -relocation-model=pic -o - %s -aarch64-min-jump-table-entries=4 -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck --check-prefix=CHECK-PIC %s
; RUN: llc -no-integrated-as -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -relocation-model=pic -aarch64-min-jump-table-entries=4 -aarch64-enable-atomic-cfg-tidy=0 -o - %s | FileCheck --check-prefix=CHECK-PIC %s
; RUN: llc -no-integrated-as -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios -aarch64-min-jump-table-entries=4 -aarch64-enable-atomic-cfg-tidy=0 | FileCheck --check-prefix=CHECK-IOS %s
; RUN: llc -no-integrated-as -code-model=tiny -verify-machineinstrs -o - %s -aarch64-min-jump-table-entries=4 -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck --check-prefix=CHECK-TINY %s

define i32 @test_jumptable(i32 %in) {
; CHECK: test_jumptable

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK-LABEL: test_jumptable:
; CHECK:     adrp [[JTPAGE:x[0-9]+]], .LJTI0_0
; CHECK:     add x[[JT:[0-9]+]], [[JTPAGE]], {{#?}}:lo12:.LJTI0_0
; CHECK:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK:     ldrb w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}]
; CHECK:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK:     br [[DEST]]

; CHECK-LARGE:     movz x[[JTADDR:[0-9]+]], #:abs_g0_nc:.LJTI0_0
; CHECK-LARGE:     movk x[[JTADDR]], #:abs_g1_nc:.LJTI0_0
; CHECK-LARGE:     movk x[[JTADDR]], #:abs_g2_nc:.LJTI0_0
; CHECK-LARGE:     movk x[[JTADDR]], #:abs_g3:.LJTI0_0
; CHECK-LARGE:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK-LARGE:     ldrb w[[OFFSET:[0-9]+]], [x[[JTADDR]], {{x[0-9]+}}]
; CHECK-LARGE:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK-LARGE:     br [[DEST]]

; CHECK-PIC-LABEL: test_jumptable:
; CHECK-PIC:     adrp [[JTPAGE:x[0-9]+]], .LJTI0_0
; CHECK-PIC:     add x[[JT:[0-9]+]], [[JTPAGE]], {{#?}}:lo12:.LJTI0_0
; CHECK-PIC:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK-PIC:     ldrb w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}]
; CHECK-PIC:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK-PIC:     br [[DEST]]

; CHECK-IOS:     adrp [[JTPAGE:x[0-9]+]], LJTI0_0@PAGE
; CHECK-IOS:     add x[[JT:[0-9]+]], [[JTPAGE]], LJTI0_0@PAGEOFF
; CHECK-IOS:     adr [[PCBASE:x[0-9]+]], [[JTBASE:LBB[0-9]+_[0-9]+]]
; CHECK-IOS:     ldrb w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}]
; CHECK-IOS:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK-IOS: br [[DEST]]

; CHECK-TINY-LABEL: test_jumptable:
; CHECK-TINY:     adr x[[JT:[0-9]+]], .LJTI0_0
; CHECK-TINY:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK-TINY:     ldrb w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}]
; CHECK-TINY:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK-TINY:     br [[DEST]]


def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}

; CHECK: .rodata

; CHECK: .LJTI0_0:
; CHECK-NEXT: .byte ([[JTBASE]]-[[JTBASE]])>>2
; CHECK-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2

define i32 @test_jumptable16(i32 %in) {

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK-LABEL: test_jumptable16:
; CHECK:     adrp [[JTPAGE:x[0-9]+]], .LJTI1_0
; CHECK:     add x[[JT:[0-9]+]], [[JTPAGE]], {{#?}}:lo12:.LJTI1_0
; CHECK:     adr [[PCBASE:x[0-9]+]], [[JTBASE:.LBB[0-9]+_[0-9]+]]
; CHECK:     ldrh w[[OFFSET:[0-9]+]], [x[[JT]], {{x[0-9]+}}, lsl #1]
; CHECK:     add [[DEST:x[0-9]+]], [[PCBASE]], x[[OFFSET]], lsl #2
; CHECK:     br [[DEST]]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  call i64 @llvm.aarch64.space(i32 1024, i64 undef)
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}

; CHECK:      .rodata
; CHECK:      .p2align 1
; CHECK: .LJTI1_0:
; CHECK-NEXT: .hword ([[JTBASE]]-[[JTBASE]])>>2
; CHECK-NEXT: .hword (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .hword (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .hword (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-NEXT: .hword (.LBB{{.*}}-[[JTBASE]])>>2

; CHECK-PIC-NOT: .data_region
; CHECK-PIC-NOT: .LJTI0_0
; CHECK-PIC: .LJTI0_0:
; CHECK-PIC-NEXT: .byte ([[JTBASE]]-[[JTBASE]])>>2
; CHECK-PIC-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-PIC-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-PIC-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-PIC-NEXT: .byte (.LBB{{.*}}-[[JTBASE]])>>2
; CHECK-PIC-NOT: .end_data_region

; CHECK-IOS: .section __TEXT,__const
; CHECK-IOS-NOT: .data_region
; CHECK-IOS: LJTI0_0:
; CHECK-IOS-NEXT:     .byte ([[JTBASE]]-[[JTBASE]])>>2
; CHECK-IOS-NEXT:     .byte (LBB{{.*}}-[[JTBASE]])>>2
; CHECK-IOS-NEXT:     .byte (LBB{{.*}}-[[JTBASE]])>>2
; CHECK-IOS-NEXT:     .byte (LBB{{.*}}-[[JTBASE]])>>2
; CHECK-IOS-NEXT:     .byte (LBB{{.*}}-[[JTBASE]])>>2
; CHECK-IOS-NOT: .end_data_region

; Compressing just the first table has the opportunity to truncate the vector of
; sizes. Make sure it doesn't.
define i32 @test_twotables(i32 %in1, i32 %in2) {
; CHECK-LABEL: test_twotables:
; CHECK: .LJTI2_0
; CHECK: .LJTI2_1

  switch i32 %in1, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  switch i32 %in1, label %def [
    i32 0, label %lbl5
    i32 1, label %lbl6
    i32 2, label %lbl7
    i32 4, label %lbl8
  ]

lbl5:
  call i64 @llvm.aarch64.space(i32 262144, i64 undef)
  ret i32 1

lbl6:
  call i64 @llvm.aarch64.space(i32 262144, i64 undef)
  ret i32 2

lbl7:
  call i64 @llvm.aarch64.space(i32 262144, i64 undef)
  ret i32 4
lbl8:
  call i64 @llvm.aarch64.space(i32 262144, i64 undef)
  ret i32 8

}

declare i64 @llvm.aarch64.space(i32, i64)

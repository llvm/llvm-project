; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; RUN: llc -mtriple=thumbv6m-eabi -mattr=+execute-only %s -o - -filetype=obj | \
; RUN:   llvm-objdump -d --no-leading-addr --no-show-raw-insn - | FileCheck %s

define void @test1() {
; CHECK-LABEL: <test1>:
;; are we using correct prologue immediate materialization pattern for
;; execute only
; CHECK: sub     sp, #0x100
%tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; CHECK-LABEL: <test2>:
;; are we using correct prologue immediate materialization pattern for
;; execute-only
; CHECK:      movs    [[REG:r[0-9]+]], #0xff
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xff
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xef
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xb8
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
;; are we using correct prologue immediate materialization pattern for
;; execute-only
; CHECK-LABEL: <test3>:
; CHECK: movs [[REG:r[0-9]+]], #0xcf
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xff
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xff
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x8
; CHECK-NEXT: adds    [[REG]], #0xf0
	%retval = alloca i32, align 4
	%tmp = alloca i32, align 4
	%a = alloca [u0x30000001 x i8], align 16
	store i32 0, ptr %tmp
;; are we choosing correct store/tSTRspi pattern for execute-only
; CHECK:      movs    [[REG:r[0-9]+]], #0x30
; CHECK-NEXT: lsls    [[REG]], [[REG]], #0x18
; CHECK-NEXT: adds    [[REG]], #0x8
	%tmp1 = load i32, ptr %tmp
        ret i32 %tmp1
}

; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=LINUX
; PR4344
; PR4416

define ptr @t() nounwind {
entry:
; DARWIN-LABEL: t:
; DARWIN: mov r0, r7

; LINUX-LABEL: t:
; LINUX: mov r0, r11
	%0 = call ptr @llvm.frameaddress(i32 0)
        ret ptr %0
}

declare ptr @llvm.frameaddress(i32) nounwind readnone

; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=LINUX
; PR4344
; PR4416

define i8* @t() nounwind {
entry:
; DARWIN: t:
; DARWIN: mov r0, r7

; LINUX: t:
; LINUX: mov r0, r11
	%0 = call i8* @llvm.frameaddress(i32 0)
        ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone

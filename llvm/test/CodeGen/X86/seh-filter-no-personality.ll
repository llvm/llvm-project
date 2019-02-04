; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s

; Mostly make sure that llvm.eh.recoverfp doesn't crash if the parent
; function lacks a personality.

declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.eh.recoverfp(i8*, i8*)

define i32 @main() {
entry:
  ret i32 0
}

define internal i32 @"filt$main"() {
entry:
  %ebp = tail call i8* @llvm.frameaddress(i32 1)
  %parentfp = tail call i8* @llvm.eh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %ebp)
  %info.addr = getelementptr inbounds i8, i8* %ebp, i32 -20
  %0 = bitcast i8* %info.addr to i32***
  %1 = load i32**, i32*** %0, align 4
  %2 = load i32*, i32** %1, align 4
  %3 = load i32, i32* %2, align 4
  %matches = icmp eq i32 %3, u0xC0000005
  %r = zext i1 %matches to i32
  ret i32 %r
}

; CHECK: _main:
; CHECK: xorl %eax, %eax
; CHECK: retl

; CHECK: _filt$main:
; CHECK: retl

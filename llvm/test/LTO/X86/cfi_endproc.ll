; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s -check-prefix=NOEXPORT
; RUN: llvm-lto -o %t3 -exported-symbol=main %t1
; RUN: llvm-nm %t3 | FileCheck %s -check-prefix=EXPORT

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".text"
module asm ".align 16, 0x90"
module asm ".type PR14512, @function"
module asm "PR14512:.cfi_startproc"
module asm "ret"
module asm ".cfi_endproc"

declare void @PR14512()

; Without -exported-symbol, main should be eliminated by LTO.
; With -exported-symbol=main, main should be preserved by LTO.
define i32 @main(i32 %argc, ptr %argv) {
; NOEXPORT-NOT: main
; EXPORT: main
  call void @PR14512()
  ret i32 0
}

; RUN: llvm-lto -o %t -dso-symbol=zed1 -dso-symbol=zed2 %t1 -O0
; RUN: llvm-nm %t | FileCheck %s -check-prefix=ZED1_AND_ZED2
; ZED1_AND_ZED2: V zed1
@zed1 = linkonce_odr global i32 42
define ptr @get_zed1() {
  ret ptr @zed1
}

; ZED1_AND_ZED2: r zed2
@zed2 = linkonce_odr unnamed_addr constant i32 42

define i32 @useZed2() {
  %x = load i32, ptr @zed2
  ret i32 %x
}

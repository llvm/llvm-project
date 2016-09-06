; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s -check-prefix=NOEXPORT
; RUN: llvm-lto -o %t3 -exported-symbol=main %t1
; RUN: llvm-nm %t3 | FileCheck %s -check-prefix=EXPORT

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
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
define i32 @main(i32 %argc, i8** %argv) {
; NOEXPORT-NOT: main
; EXPORT: main
  call void @PR14512()
  ret i32 0
}

; RUN: llvm-lto -o %t -dso-symbol=zed1 -dso-symbol=zed2 %t1 -O0
; RUN: llvm-nm %t | FileCheck %s -check-prefix=ZED1_AND_ZED2
; ZED1_AND_ZED2: V zed1
@zed1 = linkonce_odr global i32 42
define i32* @get_zed1() {
  ret i32* @zed1
}

; ZED1_AND_ZED2: r zed2
@zed2 = linkonce_odr unnamed_addr constant i32 42

define i32 @useZed2() {
  %x = load i32, i32* @zed2
  ret i32 %x
}

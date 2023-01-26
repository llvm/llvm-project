; RUN: llvm-as < %s > %t
; RUN: llvm-lto -exported-symbol=_DllMain@12 -filetype=asm -o - %t | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.23918"

; CHECK: .globl _DllMain@12
define x86_stdcallcc i32 @DllMain(ptr %module, i32 %reason, ptr %reserved) {
  ret i32 1
}

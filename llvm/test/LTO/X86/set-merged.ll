; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=_main -set-merged-module -o %t2 %t1
; RUN: llvm-objdump --print-imm-hex=false -d %t2 | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK: _main
; CHECK: movl $132
define i32 @_Z3fooi(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %a.addr, align 4
  %call = call i32 @_Z4bar2i(i32 %1)
  %add = add nsw i32 %0, %call
  ret i32 %add
}

define i32 @_Z4bar2i(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %mul = mul nsw i32 2, %0
  ret i32 %mul
}

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @_Z3fooi(i32 44)
  ret i32 %call
}

; RUN: llc < %s -mtriple=mips -mattr=mips16 | FileCheck %s -check-prefix=M16
; RUN: llc < %s -mtriple=mips -mcpu=mips2 | FileCheck %s -check-prefix=GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32 | FileCheck %s -check-prefix=GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32r6 | FileCheck %s -check-prefix=GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips3 -target-abi n64 | FileCheck %s -check-prefix=GP64
; RUN: llc < %s -mtriple=mips -mcpu=mips64 -target-abi n64 | FileCheck %s -check-prefix=GP64
; RUN: llc < %s -mtriple=mips -mcpu=mips64r6 -target-abi n64 | FileCheck %s -check-prefix=GP64

declare void @bar(ptr)

define void @foo(i32 %sz) {
  ; ALL-LABEL: foo:

  ; M16-NOT:        addiu     $sp, 0 # 16 bit inst
  ; GP32-NOT:       addiu     $sp, $sp, 0
  ; GP64-NOT:       daddiu    $sp, $sp, 0
  %a = alloca i32, i32 %sz
  call void @bar(ptr %a)
  ret void
}

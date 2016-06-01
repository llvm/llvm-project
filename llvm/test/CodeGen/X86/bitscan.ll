; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7 | FileCheck %s --check-prefix=ALL --check-prefix=64-BIT
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=corei7 | FileCheck %s --check-prefix=ALL --check-prefix=32-BIT
declare i32 @llvm.x86.bit.scan.forward.32(i32 %val)
declare i32 @llvm.x86.bit.scan.reverse.32(i32 %val)

define i32 @test_bsf(i32 %val) {
  %call = call i32 @llvm.x86.bit.scan.forward.32(i32 %val)
  ret i32 %call

; ALL-LABEL: test_bsf:
; 64-BIT:    bsfl %edi, %eax
; 32-BIT:    bsfl 4(%esp), %eax
}

define i32 @test_bsr(i32 %val) {
  %call = call i32 @llvm.x86.bit.scan.reverse.32(i32 %val)
  ret i32 %call

; ALL-LABEL: test_bsr:
; 64-BIT:    bsrl %edi, %eax
; 32-BIT:    bsrl 4(%esp), %eax
}


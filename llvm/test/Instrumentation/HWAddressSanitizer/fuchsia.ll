; Check HWASan shadow mapping on Fuchsia.
; RUN: opt -passes=hwasan -S -mtriple=aarch64-unknown-fuchsia < %s | FileCheck %s

define i32 @test_load(ptr %a) sanitize_hwaddress {
; CHECK: %.hwasan.shadow = call ptr asm "", "=r,0"(ptr null)
entry:
  %x = load i32, ptr %a, align 4
  ret i32 %x
}

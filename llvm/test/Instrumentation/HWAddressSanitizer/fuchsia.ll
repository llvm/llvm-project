; Check HWASan shadow mapping on Fuchsia.
; RUN: opt -hwasan -S -mtriple=aarch64-unknown-fuchsia < %s | FileCheck %s

define i32 @test_load(i32* %a) sanitize_hwaddress {
; CHECK: %.hwasan.shadow = call i8* asm "", "=r,0"(i8* null)
entry:
  %x = load i32, i32* %a, align 4
  ret i32 %x
}

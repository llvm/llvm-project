; Check HWASan shadow mapping on Fuchsia.
; RUN: opt -passes=hwasan -S -mtriple=aarch64-unknown-fuchsia < %s | FileCheck %s

define i32 @test_load(ptr %a) sanitize_hwaddress {
; CHECK: [[SHADOW:%.*]] = load ptr, ptr @__hwasan_shadow_memory_dynamic_address, align 8
entry:
  %x = load i32, ptr %a, align 4
  ret i32 %x
}

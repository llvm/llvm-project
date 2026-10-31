; Test that MTE-tag stripping is applied only on Darwin aarch64

; RUN: opt < %s -passes=asan -S -mtriple=arm64-apple-darwin | FileCheck %s --check-prefix=DARWIN-AARCH64
; RUN: opt < %s -passes=asan -S -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefix=DARWIN-X64
; RUN: opt < %s -passes=asan -S -mtriple=aarch64-unknown-linux-gnu | FileCheck %s --check-prefix=LINUX-AARCH64

define i32 @test_load(ptr %p) sanitize_address {
entry:
  %tmp1 = load i32, ptr %p, align 4
  ret i32 %tmp1
}

; Darwin aarch64 should strip MTE-tags before shadow translation
; The mask ~(0x0f << 56) = -1080863910568919041 clears bits 56-59
; DARWIN-AARCH64-LABEL: @test_load
; DARWIN-AARCH64-NOT: ret
; DARWIN-AARCH64: and i64 {{%[0-9a-z]+}}, -1080863910568919041
; DARWIN-AARCH64: lshr i64
; DARWIN-AARCH64: {{or|add}}
; DARWIN-AARCH64: ret i32

; Darwin x86_64 should NOT strip MTE-tags (not aarch64)
; DARWIN-X64-LABEL: @test_load
; DARWIN-X64-NOT: ret
; DARWIN-X64-NOT: and i64 {{%[0-9a-z]+}}, -1080863910568919041
; DARWIN-X64: lshr i64
; DARWIN-X64: {{or|add}}
; DARWIN-X64: ret i32

; Linux aarch64 should NOT strip MTE-tags (not Darwin)
; LINUX-AARCH64-LABEL: @test_load
; LINUX-AARCH64-NOT: ret
; LINUX-AARCH64-NOT: and i64 {{%[0-9a-z]+}}, -1080863910568919041
; LINUX-AARCH64: lshr i64
; LINUX-AARCH64: {{or|add}}
; LINUX-AARCH64: ret i32

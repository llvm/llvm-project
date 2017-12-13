; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define i8 @test_load8(i8* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load8(
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #256", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i8, i8* %a, align 4
; CHECK: ret i8 %[[G]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i16 @test_load16(i16* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load16(
; CHECK: %[[A:[^ ]*]] = ptrtoint i16* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #257", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i16, i16* %a, align 4
; CHECK: ret i16 %[[G]]

entry:
  %b = load i16, i16* %a, align 4
  ret i16 %b
}

define i32 @test_load32(i32* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load32(
; CHECK: %[[A:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #258", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i32, i32* %a, align 4
; CHECK: ret i32 %[[G]]

entry:
  %b = load i32, i32* %a, align 4
  ret i32 %b
}

define i64 @test_load64(i64* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load64(
; CHECK: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #259", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i64, i64* %a, align 8
; CHECK: ret i64 %[[G]]

entry:
  %b = load i64, i64* %a, align 8
  ret i64 %b
}

define i128 @test_load128(i128* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load128(
; CHECK: %[[A:[^ ]*]] = ptrtoint i128* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #260", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i128, i128* %a, align 16
; CHECK: ret i128 %[[G]]

entry:
  %b = load i128, i128* %a, align 16
  ret i128 %b
}

define i40 @test_load40(i40* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load40(
; CHECK: %[[A:[^ ]*]] = ptrtoint i40* %a to i64
; CHECK: call void @__hwasan_load(i64 %[[A]], i64 5)
; CHECK: %[[B:[^ ]*]] = load i40, i40* %a
; CHECK: ret i40 %[[B]]

entry:
  %b = load i40, i40* %a, align 4
  ret i40 %b
}

define void @test_store8(i8* %a, i8 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store8(
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #272", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: store i8 %b, i8* %a, align 4
; CHECK: ret void

entry:
  store i8 %b, i8* %a, align 4
  ret void
}

define void @test_store16(i16* %a, i16 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store16(
; CHECK: %[[A:[^ ]*]] = ptrtoint i16* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #273", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: store i16 %b, i16* %a, align 4
; CHECK: ret void

entry:
  store i16 %b, i16* %a, align 4
  ret void
}

define void @test_store32(i32* %a, i32 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store32(
; CHECK: %[[A:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #274", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: store i32 %b, i32* %a, align 4
; CHECK: ret void

entry:
  store i32 %b, i32* %a, align 4
  ret void
}

define void @test_store64(i64* %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store64(
; CHECK: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #275", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: store i64 %b, i64* %a, align 8
; CHECK: ret void

entry:
  store i64 %b, i64* %a, align 8
  ret void
}

define void @test_store128(i128* %a, i128 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store128(
; CHECK: %[[A:[^ ]*]] = ptrtoint i128* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; CHECK: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; CHECK: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "hlt #276", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: store i128 %b, i128* %a, align 16
; CHECK: ret void

entry:
  store i128 %b, i128* %a, align 16
  ret void
}

define void @test_store40(i40* %a, i40 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store40(
; CHECK: %[[A:[^ ]*]] = ptrtoint i40* %a to i64
; CHECK: call void @__hwasan_store(i64 %[[A]], i64 5)
; CHECK: store i40 %b, i40* %a
; CHECK: ret void

entry:
  store i40 %b, i40* %a, align 4
  ret void
}

define void @test_store_unaligned(i64* %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store_unaligned(
; CHECK: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; CHECK: call void @__hwasan_store(i64 %[[A]], i64 8)
; CHECK: store i64 %b, i64* %a, align 4
; CHECK: ret void

entry:
  store i64 %b, i64* %a, align 4
  ret void
}

define i8 @test_load_noattr(i8* %a) {
; CHECK-LABEL: @test_load_noattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i8 @test_load_notmyattr(i8* %a) sanitize_address {
; CHECK-LABEL: @test_load_notmyattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i8 @test_load_addrspace(i8 addrspace(256)* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load_addrspace(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8 addrspace(256)* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8 addrspace(256)* %a, align 4
  ret i8 %b
}

; CHECK: declare void @__hwasan_init()

; CHECK:      define internal void @hwasan.module_ctor() {
; CHECK-NEXT:   call void @__hwasan_init()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

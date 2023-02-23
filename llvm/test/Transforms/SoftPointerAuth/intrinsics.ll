; RUN: opt < %s -passes=soft-ptrauth -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.__block_literal_generic = type { ptr, i32, i32, ptr, ptr }

@blockptr = common global ptr null, align 8

define internal void @test1() {
entry:
  %block = load ptr, ptr @blockptr, align 8
  %fnptr_addr = getelementptr inbounds %struct.__block_literal_generic, ptr %block, i32 0, i32 3
  %fnptr = load ptr, ptr %fnptr_addr, align 8
  %discriminator = ptrtoint ptr %fnptr_addr to i64
  call void %fnptr(ptr %block) [ "ptrauth"(i32 1, i64 %discriminator) ]
  ret void
}

; CHECK: define internal void @test1() {
; CHECK: %fnptr_addr = getelementptr inbounds %struct.__block_literal_generic, ptr %block, i32 0, i32 3
; CHECK-NEXT: %fnptr = load ptr, ptr %fnptr_addr, align 8
; CHECK-NEXT: %discriminator = ptrtoint ptr %fnptr_addr to i64
; CHECK-NEXT: [[FNPTR_AUTH:%.*]] = call ptr @__ptrauth_auth(ptr %fnptr, i32 1, i64 %discriminator) [[NOUNWIND:#[0-9]+]]
; CHECK-NEXT: call void [[FNPTR_AUTH]](ptr %block){{$}}

; CHECK: attributes [[NOUNWIND]] = { nounwind }

; RUN: opt < %s -soft-ptrauth -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { i8*, i32, i32, i8*, %struct.__block_descriptor* }

@blockptr = common global void ()* null, align 8

define internal void @test1() {
entry:
  %0 = load void ()*, void ()** @blockptr, align 8
  %block = bitcast void ()* %0 to %struct.__block_literal_generic*
  %fnptr_addr = getelementptr inbounds %struct.__block_literal_generic, %struct.__block_literal_generic* %block, i32 0, i32 3
  %block_opaque = bitcast %struct.__block_literal_generic* %block to i8*
  %1 = load i8*, i8** %fnptr_addr, align 8
  %fnptr = bitcast i8* %1 to void (i8*)*
  %discriminator = ptrtoint i8** %fnptr_addr to i64
  call void %fnptr(i8* %block_opaque) [ "ptrauth"(i32 1, i64 %discriminator) ]
  ret void
}

; CHECK: define internal void @test1() {
; CHECK:      %fnptr_addr = getelementptr inbounds %struct.__block_literal_generic, %struct.__block_literal_generic* %block, i32 0, i32 3
; CHECK-NEXT: %block_opaque = bitcast %struct.__block_literal_generic* %block to i8*
; CHECK-NEXT: [[T0:%.*]] = load i8*, i8** %fnptr_addr, align 8
; CHECK-NEXT: %fnptr = bitcast i8* [[T0]] to void (i8*)*
; CHECK-NEXT: %discriminator = ptrtoint i8** %fnptr_addr to i64
; CHECK-NEXT: [[FNPTR_CAST:%.*]] = bitcast void (i8*)* %fnptr to i8*
; CHECK-NEXT: [[FNPTR_AUTH:%.*]] = call i8* @__ptrauth_auth(i8* [[FNPTR_CAST]], i32 1, i64 %discriminator) [[NOUNWIND:#[0-9]+]]
; CHECK-NEXT: [[FNPTR_AUTH_CAST:%.*]] = bitcast i8* [[FNPTR_AUTH]] to void (i8*)*
; CHECK-NEXT: call void [[FNPTR_AUTH_CAST]](i8* %block_opaque){{$}}

; CHECK: attributes [[NOUNWIND]] = { nounwind }

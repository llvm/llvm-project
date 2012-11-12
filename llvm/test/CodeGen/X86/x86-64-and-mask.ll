; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

; This should be a single mov, not a load of immediate + andq.
; CHECK: test:
; CHECK: movl %edi, %eax

define i64 @test(i64 %x) nounwind {
entry:
	%tmp123 = and i64 %x, 4294967295		; <i64> [#uses=1]
	ret i64 %tmp123
}

; This copy can't be coalesced away because it needs the implicit zero-extend.
; CHECK: bbb:
; CHECK: movl %edi, %edi

define void @bbb(i64 %x) nounwind {
  %t = and i64 %x, 4294967295
  call void @foo(i64 %t)
  ret void
}

; This should use a 32-bit and with implicit zero-extension, not a 64-bit and
; with a separate mov to materialize the mask.
; rdar://7527390
; CHECK: ccc:
; CHECK: andl $-1048593, %edi

declare void @foo(i64 %x) nounwind

define void @ccc(i64 %x) nounwind {
  %t = and i64 %x, 4293918703
  call void @foo(i64 %t)
  ret void
}

; This requires a mov and a 64-bit and.
; CHECK: ddd:
; CHECK: movabsq $4294967296, %r
; CHECK: andq %rax, %rdi

define void @ddd(i64 %x) nounwind {
  %t = and i64 %x, 4294967296
  call void @foo(i64 %t)
  ret void
}

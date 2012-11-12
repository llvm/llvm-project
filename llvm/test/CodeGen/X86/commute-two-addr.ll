; The register allocator can commute two-address instructions to avoid
; insertion of register-register copies.

; Make sure there are only 3 mov's for each testcase
; RUN: llc < %s -mtriple=i686-pc-linux-gnu   | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s -check-prefix=DARWIN


@G = external global i32                ; <i32*> [#uses=2]

declare void @ext(i32)

define i32 @t1(i32 %X, i32 %Y) nounwind {
; LINUX: t1:
; LINUX: movl 4(%esp), %eax
; LINUX: movl 8(%esp), %ecx
; LINUX: addl %eax, %ecx
; LINUX: movl %ecx, G
        %Z = add i32 %X, %Y             ; <i32> [#uses=1]
        store i32 %Z, i32* @G
        ret i32 %X
}

define i32 @t2(i32 %X, i32 %Y) nounwind {
; LINUX: t2:
; LINUX: movl 4(%esp), %eax
; LINUX: movl 8(%esp), %ecx
; LINUX: xorl %eax, %ecx
; LINUX: movl %ecx, G
        %Z = xor i32 %X, %Y             ; <i32> [#uses=1]
        store i32 %Z, i32* @G
        ret i32 %X
}

; rdar://8762995
%0 = type { i64, i32 }

define %0 @t3(i32 %lb, i8 zeroext %has_lb, i8 zeroext %lb_inclusive, i32 %ub, i8 zeroext %has_ub, i8 zeroext %ub_inclusive) nounwind {
entry:
; DARWIN: t3:
; DARWIN: shll $16
; DARWIN: shlq $32, %rcx
; DARWIN-NOT: leaq
; DARWIN: orq %rcx, %rax
  %tmp21 = zext i32 %lb to i64
  %tmp23 = zext i32 %ub to i64
  %tmp24 = shl i64 %tmp23, 32
  %ins26 = or i64 %tmp24, %tmp21
  %tmp28 = zext i8 %has_lb to i32
  %tmp33 = zext i8 %has_ub to i32
  %tmp34 = shl i32 %tmp33, 8
  %tmp38 = zext i8 %lb_inclusive to i32
  %tmp39 = shl i32 %tmp38, 16
  %tmp43 = zext i8 %ub_inclusive to i32
  %tmp44 = shl i32 %tmp43, 24
  %ins31 = or i32 %tmp39, %tmp28
  %ins36 = or i32 %ins31, %tmp34
  %ins46 = or i32 %ins36, %tmp44
  %tmp16 = insertvalue %0 undef, i64 %ins26, 0
  %tmp19 = insertvalue %0 %tmp16, i32 %ins46, 1
  ret %0 %tmp19
}

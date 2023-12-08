; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck -check-prefix CHK %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; CHK-NOT: InlineAsm

; CHECK-LABEL: foo:
; CHECK: bswapq
define i64 @foo(i64 %x) nounwind {
	%asmtmp = tail call i64 asm "bswap $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
	ret i64 %asmtmp
}

; CHECK-LABEL: bar:
; CHECK: bswapq
define i64 @bar(i64 %x) nounwind {
	%asmtmp = tail call i64 asm "bswapq ${0:q}", "=r,0,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
	ret i64 %asmtmp
}

; CHECK-LABEL: pen:
; CHECK: bswapl
define i32 @pen(i32 %x) nounwind {
	%asmtmp = tail call i32 asm "bswapl ${0:q}", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %x) nounwind
	ret i32 %asmtmp
}

; CHECK-LABEL: s16:
; CHECK: rolw    $8,
define zeroext i16 @s16(i16 zeroext %x) nounwind {
  %asmtmp = tail call i16 asm "rorw $$8, ${0:w}", "=r,0,~{dirflag},~{fpsr},~{flags},~{cc}"(i16 %x) nounwind
  ret i16 %asmtmp
}

; CHECK-LABEL: t16:
; CHECK: rolw    $8,
define zeroext i16 @t16(i16 zeroext %x) nounwind {
  %asmtmp = tail call i16 asm "rorw $$8, ${0:w}", "=r,0,~{cc},~{dirflag},~{fpsr},~{flags}"(i16 %x) nounwind
  ret i16 %asmtmp
}

; CHECK-LABEL: u16:
; CHECK: rolw    $8,
define zeroext i16 @u16(i16 zeroext %x) nounwind {
  %asmtmp = tail call i16 asm "rolw $$8, ${0:w}", "=r,0,~{dirflag},~{fpsr},~{flags},~{cc}"(i16 %x) nounwind
  ret i16 %asmtmp
}

; CHECK-LABEL: v16:
; CHECK: rolw    $8,
define zeroext i16 @v16(i16 zeroext %x) nounwind {
  %asmtmp = tail call i16 asm "rolw $$8, ${0:w}", "=r,0,~{cc},~{dirflag},~{fpsr},~{flags}"(i16 %x) nounwind
  ret i16 %asmtmp
}

; CHECK-LABEL: s32:
; CHECK: bswapl
define i32 @s32(i32 %x) nounwind {
  %asmtmp = tail call i32 asm "bswap $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %x) nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: t32:
; CHECK: bswapl
define i32 @t32(i32 %x) nounwind {
  %asmtmp = tail call i32 asm "bswap $0", "=r,0,~{dirflag},~{flags},~{fpsr}"(i32 %x) nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: u32:
; CHECK: bswapl
define i32 @u32(i32 %x) nounwind {
  %asmtmp = tail call i32 asm "rorw $$8, ${0:w};rorl $$16, $0;rorw $$8, ${0:w}", "=r,0,~{cc},~{dirflag},~{flags},~{fpsr}"(i32 %x) nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: s64:
; CHECK: bswapq
define i64 @s64(i64 %x) nounwind {
  %asmtmp = tail call i64 asm "bswap ${0:q}", "=r,0,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
  ret i64 %asmtmp
}

; CHECK-LABEL: t64:
; CHECK: bswapq
define i64 @t64(i64 %x) nounwind {
  %asmtmp = tail call i64 asm "bswap ${0:q}", "=r,0,~{fpsr},~{dirflag},~{flags}"(i64 %x) nounwind
  ret i64 %asmtmp
}

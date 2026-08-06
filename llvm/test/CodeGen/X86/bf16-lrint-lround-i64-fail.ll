; RUN: not --crash llc -o /dev/null %s -mtriple=i686-linux-gnu -mattr=+sse2 2>&1 | FileCheck %s
; RUN: not --crash llc -o /dev/null %s -mtriple=i686-linux-gnu 2>&1 | FileCheck %s

; The i64-result forms of the four new bf16 LRINT/LLRINT/LROUND/LLROUND Expand
; actions (X86ISelLowering.cpp:693-696) crash on 32-bit X86:
;
;   LLVM ERROR: unsupported library call operation
;
; This is bf16-specific, and it is a regression rather than a shared 16-bit-float
; limitation. Verified on this branch's own llc:
;
;   i686 +sse2   lrint i64 <- bfloat   CRASH
;   i686 +sse2   lrint i64 <- half     ok (__extendhfsf2 then lrintf)
;   i686 +sse2   lrint i64 <- float    ok (lrintf)
;   x86_64       lrint i64 <- bfloat   ok
;   i686 +sse2   lrint i32 <- bfloat   ok
;
; So it needs all three of: 32-bit target, i64 result, bf16 operand. It also
; fails for the strict form (llvm.experimental.constrained.lrint.i64.bf16).
;
; Mechanism: on 32-bit with X87, LRINT/LLRINT to i64 are marked Custom
; (X86ISelLowering.cpp:305-308), and f16 satisfies that by Expanding through an
; f32 libcall first. bf16 has FP_EXTEND set to Custom instead, so the bf16
; operand does not get turned into the f32 form the i64 LRINT path expects, and
; legalization reaches makeLibCall with RTLIB::Unsupported. Note the branch
; added the four non-strict bf16 Expand lines but not the four STRICT_* ones
; that f16 has right below them at lines 766-769, which is a related asymmetry.
;
; FIXME: this should not crash. The i32-result forms that already work are
; covered by bfloat-lrint-lround.ll, so any fix can be checked against those.

; CHECK: LLVM ERROR: unsupported library call operation
define i64 @lrint_i64_bf16(bfloat %a) nounwind {
  %r = call i64 @llvm.lrint.i64.bf16(bfloat %a)
  ret i64 %r
}

define i64 @llrint_bf16(bfloat %a) nounwind {
  %r = call i64 @llvm.llrint.i64.bf16(bfloat %a)
  ret i64 %r
}

define i64 @lround_i64_bf16(bfloat %a) nounwind {
  %r = call i64 @llvm.lround.i64.bf16(bfloat %a)
  ret i64 %r
}

define i64 @llround_bf16(bfloat %a) nounwind {
  %r = call i64 @llvm.llround.i64.bf16(bfloat %a)
  ret i64 %r
}

declare i64 @llvm.lrint.i64.bf16(bfloat)
declare i64 @llvm.llrint.i64.bf16(bfloat)
declare i64 @llvm.lround.i64.bf16(bfloat)
declare i64 @llvm.llround.i64.bf16(bfloat)

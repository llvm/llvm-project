; RUN: llc -o /dev/null %s -mtriple=i686-linux-gnu -mattr=+sse2 2>&1 | FileCheck %s
; RUN: llc -o /dev/null %s -mtriple=i686-linux-gnu 2>&1 | FileCheck %s
XFAIL: *

; The i64-result forms of the four new bf16 LRINT/LLRINT/LROUND/LLROUND Expand
; actions (X86ISelLowering.cpp:693-696) crash on 32-bit X86:
; FIXME: this should not crash. The i32-result forms that already work are
; covered by bf16-lrint-lround.ll, so any fix can be checked against those.

; CHECK-NOT: {{Unexpected lrint input type!|unsupported library call operation}}
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

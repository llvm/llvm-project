; RUN: llc %s -mtriple=sparc -o - | FileCheck %s
; RUN: llc %s -mtriple=sparc64 -o - | FileCheck %s
declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128)
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)

define i32 @mul(i128 %a, i128 %b, i128* %r) {
; CHECK-LABEL: mul
; CHECK-NOT: call __muloti4
  %mul4 = tail call { i128, i1 } @llvm.smul.with.overflow.i128(i128 %a, i128 %b)
  %mul.val = extractvalue { i128, i1 } %mul4, 0
  %mul.ov = extractvalue { i128, i1 } %mul4, 1
  %mul.not.ov = xor i1 %mul.ov, true
  store i128 %mul.val, i128* %r, align 16
  %conv = zext i1 %mul.not.ov to i32
  ret i32 %conv
}

define i32 @mul2(i64 %a, i64 %b, i64* %r) {
; CHECK-LABEL: mul2
; CHECK-NOT: call __mulodi4
  %mul4 = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %mul.val = extractvalue { i64, i1 } %mul4, 0
  %mul.ov = extractvalue { i64, i1 } %mul4, 1
  %mul.not.ov = xor i1 %mul.ov, true
  store i64 %mul.val, i64* %r, align 16
  %conv = zext i1 %mul.not.ov to i32
  ret i32 %conv
}

; RUN: llc < %s -mtriple=mips64el -mattr=+soft-float | FileCheck %s

; fp128 llrint is only available on mips64, where long double is fp128 and the
; `l`-suffixed call is the correct-width fp128 libcall.

define signext i32 @testmswl(fp128 %x) {
; CHECK-LABEL: testmswl:
; CHECK:       jal     llrintl
entry:
  %0 = tail call i64 @llvm.llrint.i64.f128(fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define i64 @testmsll(fp128 %x) {
; CHECK-LABEL: testmsll:
; CHECK:       jal     llrintl
entry:
  %0 = tail call i64 @llvm.llrint.i64.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.llrint.i64.f128(fp128) nounwind readnone

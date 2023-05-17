;; Source
;; int square(unsigned short a) {
;;   return a * a;
;; }
;; Command
;; clang -cc1 -triple spir -emit-llvm -O2 -o NoSignedUnsignedWrap.ll test.cl
;;
;; Positive tests:
;;
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NEGATIVE
;;
;; Negative tests:
;;
;; Check that backend is able to skip nsw/nuw attributes if extension is
;; disabled implicitly or explicitly and if max SPIR-V version is lower then 1.4

; CHECK-SPIRV-DAG: OpDecorate %[[#]] NoSignedWrap
; CHECK-SPIRV-DAG: OpDecorate %[[#]] NoUnsignedWrap
;
; CHECK-SPIRV-NEGATIVE-NOT: OpExtension "SPV_KHR_no_integer_wrap_decoration"
; CHECK-SPIRV-NEGATIVE-NOT: OpDecorate %[[#]] NoSignedWrap
; CHECK-SPIRV-NEGATIVE-NOT: OpDecorate %[[#]] NoUnsignedWrap

define spir_func i32 @square(i16 zeroext %a) local_unnamed_addr {
entry:
  %conv = zext i16 %a to i32
  %mul = mul nuw nsw i32 %conv, %conv
  ret i32 %mul
}

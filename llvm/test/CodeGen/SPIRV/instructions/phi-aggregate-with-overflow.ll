; A `with.overflow` result is a multi-register aggregate that is never rewritten
; to a value-id, but PHI is, so verify there is no type mismatch in the PHI.
; See https://github.com/llvm/llvm-project/issues/203586.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I64]] %[[#BOOL]]

; CHECK: %[[#PHI:]] = OpPhi %[[#STRUCT]]
; CHECK: OpCompositeExtract %[[#I64]] %[[#PHI]] 0
; CHECK: OpCompositeExtract %[[#BOOL]] %[[#PHI]] 1

define i64 @phi_overflow(i64 %a, i64 %b, i1 %c) {
entry:
  br i1 %c, label %bb0, label %bb1

bb0:
  %0 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  br label %epilog

bb1:
  %1 = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  br label %epilog

epilog:
  %2 = phi { i64, i1 } [ %0, %bb0 ], [ %1, %bb1 ]
  %3 = extractvalue { i64, i1 } %2, 0
  %4 = extractvalue { i64, i1 } %2, 1
  %5 = zext i1 %4 to i64
  %6 = add i64 %3, %5
  ret i64 %6
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64)

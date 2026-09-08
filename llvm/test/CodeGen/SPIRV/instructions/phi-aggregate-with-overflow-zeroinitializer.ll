; A constant aggregate PHI arm (zeroinitializer) is already a value-id (a shared
; OpConstantNull), so it must coexist with reassembled multi-register arms in
; the same PHI. See https://github.com/llvm/llvm-project/issues/203586.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I64]] %[[#BOOL]]
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#STRUCT]]

; CHECK: %[[#PHI:]] = OpPhi %[[#STRUCT]] {{.*}}%[[#NULL]]
; CHECK: OpCompositeExtract %[[#I64]] %[[#PHI]] 0
define i64 @phi_overflow_zeroinitializer(i64 %a, i64 %b, i1 %c) {
entry:
  br i1 %c, label %bb0, label %bb1

bb0:
  %0 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  br label %epilog

bb1:
  br label %epilog

epilog:
  %1 = phi { i64, i1 } [ %0, %bb0 ], [ zeroinitializer, %bb1 ]
  %2 = extractvalue { i64, i1 } %1, 0
  %3 = extractvalue { i64, i1 } %1, 1
  %4 = zext i1 %3 to i64
  %5 = add i64 %2, %4
  ret i64 %5
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

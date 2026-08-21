; A select or freeze of an aggregate is lowered to a single value-id just like a
; PHI, so a multi-register `with.overflow` arm must likewise be reassembled into
; a composite. See https://github.com/llvm/llvm-project/issues/203586.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I64]] %[[#BOOL]]

; CHECK: OpSelect %[[#STRUCT]]
define i64 @select_overflow(i64 %a, i64 %b, i1 %c) {
  %add = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %sub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %sel = select i1 %c, { i64, i1 } %add, { i64, i1 } %sub
  %v = extractvalue { i64, i1 } %sel, 0
  %o = extractvalue { i64, i1 } %sel, 1
  %z = zext i1 %o to i64
  %r = add i64 %v, %z
  ret i64 %r
}

define i64 @freeze_overflow(i64 %a, i64 %b) {
  %mul = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %a, i64 %b)
  %frz = freeze { i64, i1 } %mul
  %v = extractvalue { i64, i1 } %frz, 0
  %o = extractvalue { i64, i1 } %frz, 1
  %z = zext i1 %o to i64
  %r = add i64 %v, %z
  ret i64 %r
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

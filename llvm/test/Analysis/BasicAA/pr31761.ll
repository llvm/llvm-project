; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.blam = type { i32, i32 }


; CHECK-DAG: MayAlias: i32* %tmp, i32* %tmp3

define i1 @ham(%struct.blam* %arg)  {
  %isNull = icmp eq %struct.blam* %arg, null
  %tmp = getelementptr  %struct.blam, %struct.blam* %arg, i64 0, i32 0
  %tmp2 = getelementptr  %struct.blam, %struct.blam* %arg, i64 0, i32 1
  %select = select i1 %isNull, i32* null, i32* %tmp2
  %tmp3 = getelementptr  i32, i32* %select, i32 -1
  load i32, i32* %tmp
  load i32, i32* %tmp3
  ret i1 true
}

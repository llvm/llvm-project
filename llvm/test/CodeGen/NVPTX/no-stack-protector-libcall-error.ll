; RUN: not opt -disable-output -mtriple=nvptx64-- -enable-selectiondag-sp=0 -passes=stack-protector %s 2>&1 | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; CHECK: error: no libcall available for stack protector
define void @func() sspreq nounwind {
  %alloca = alloca i32, align 4
  call void @capture(ptr %alloca)
  ret void
}

declare void @capture(ptr)

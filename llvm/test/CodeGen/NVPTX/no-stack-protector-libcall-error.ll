; RUN: not opt -disable-output -mtriple=nvptx64-- -enable-selectiondag-sp=0 -passes='require<libcall-lowering-info>,stack-protector' %s 2>&1 | FileCheck %s

; CHECK: error: no libcall available for stack protector
define void @func() sspreq nounwind {
  %alloca = alloca i32, align 4
  call void @capture(ptr %alloca)
  ret void
}

declare void @capture(ptr)

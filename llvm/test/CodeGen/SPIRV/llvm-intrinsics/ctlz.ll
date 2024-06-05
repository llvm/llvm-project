; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] clz
; CHECK: OpFunctionEnd

define spir_func i32 @TestClz(i32 %x) local_unnamed_addr {
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %0
}

declare i32 @llvm.ctlz.i32(i32, i1 immarg)

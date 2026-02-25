; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s


; Legalizing for load from a pointer PHI

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FTYPE:]] = OpTypeFunction %[[#INT]]
; CHECK-DAG: %[[#PTR:]] = OpTypePointer StorageBuffer %[[#INT]]
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#PTR]]
; CHECK-DAG: %[[#FALSE:]] = OpConstantFalse %[[#BOOL]]
; CHECK:     %[[#]] = OpFunction %[[#INT]] None %[[#FTYPE]]
; CHECK:     %[[#]] = OpLabel
; CHECK:     OpSelectionMerge %[[#MERGE:]] None
; CHECK:     OpBranchConditional %[[#FALSE]] %[[#]] %[[#]]
; CHECK:     %[[#]] = OpLabel
; CHECK:     OpBranch %[[#MERGE]]
; CHECK:     %[[#]] = OpLabel
; CHECK:     OpBranch %[[#MERGE]]
; CHECK-NEXT:     %[[#MERGE]] = OpLabel
; CHECK-NEXT:     %[[#RES:]] = OpLoad %[[#INT]] %[[#NULL]]
; CHECK-NEXT:     OpReturnValue %[[#RES]]
; CHECK:     OpFunctionEnd

define i32 @main() {
entry:
  br i1 false, label %sw.epilog23.i.sink.split, label %sw.epilog.i.thread

sw.epilog.i.thread:                               ; preds = %entry
  br label %sw.epilog23.i.sink.split

sw.epilog23.i.sink.split:                         ; preds = %sw.epilog.i.thread, %entry
  %.sink = phi ptr addrspace(11) [ null, %sw.epilog.i.thread ], [ null, %entry ]
  %0 = load i32, ptr addrspace(11) %.sink, align 4
  ret i32 %0
}

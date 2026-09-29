; RUN: opt -S -aa-pipeline=objc-arc-aa,basic-aa -passes=gvn < %s | FileCheck %s
; rdar://10050579

; objc_retainBlock stores into %repeater so the load from after the
; call isn't forwardable from the store before the call.

; CHECK: %tmp16 = call ptr @llvm.objc.retainBlock(ptr %block) [[NUW:#[0-9]+]]
; CHECK: %tmp18 = load ptr, ptr %byref.forwarding, align 8
; CHECK: %repeater12 = getelementptr inbounds %struct.__block_byref_repeater, ptr %tmp18, i64 0, i32 6
; CHECK: store ptr %tmp16, ptr %repeater12, align 8

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%0 = type opaque
%struct.__block_byref_repeater = type { ptr, ptr, i32, i32, ptr, ptr, ptr }
%struct.__block_descriptor = type { i64, i64 }

define void @foo() noreturn {
entry:
  %repeater = alloca %struct.__block_byref_repeater, align 8
  %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_repeater, ptr %repeater, i64 0, i32 1
  %tmp10 = getelementptr inbounds %struct.__block_byref_repeater, ptr %repeater, i64 0, i32 6
  store ptr null, ptr %tmp10, align 8
  %block.captured11 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, ptr %block, i64 0, i32 6
  store ptr %repeater, ptr %block.captured11, align 8
  %tmp16 = call ptr @llvm.objc.retainBlock(ptr %block) nounwind
  %tmp18 = load ptr, ptr %byref.forwarding, align 8
  %repeater12 = getelementptr inbounds %struct.__block_byref_repeater, ptr %tmp18, i64 0, i32 6
  %tmp13 = load ptr, ptr %repeater12, align 8
  store ptr %tmp16, ptr %repeater12, align 8
  ret void
}

declare ptr @llvm.objc.retainBlock(ptr)

; CHECK: attributes #0 = { noreturn }
; CHECK: attributes [[NUW]] = { nounwind }

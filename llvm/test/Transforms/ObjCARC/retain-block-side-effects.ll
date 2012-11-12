; RUN: opt -S -objc-arc-aa -basicaa -gvn < %s | FileCheck %s
; rdar://10050579

; objc_retainBlock stores into %repeater so the load from after the
; call isn't forwardable from the store before the call.

; CHECK: %tmp16 = call i8* @objc_retainBlock(i8* %tmp15) nounwind
; CHECK: %tmp17 = bitcast i8* %tmp16 to void ()*
; CHECK: %tmp18 = load %struct.__block_byref_repeater** %byref.forwarding, align 8
; CHECK: %repeater12 = getelementptr inbounds %struct.__block_byref_repeater* %tmp18, i64 0, i32 6
; CHECK: store void ()* %tmp17, void ()** %repeater12, align 8

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%0 = type opaque
%struct.__block_byref_repeater = type { i8*, %struct.__block_byref_repeater*, i32, i32, i8*, i8*, void ()* }
%struct.__block_descriptor = type { i64, i64 }

define void @foo() noreturn {
entry:
  %repeater = alloca %struct.__block_byref_repeater, align 8
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i8* }>, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_repeater* %repeater, i64 0, i32 1
  %tmp10 = getelementptr inbounds %struct.__block_byref_repeater* %repeater, i64 0, i32 6
  store void ()* null, void ()** %tmp10, align 8
  %block.captured11 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i8* }>* %block, i64 0, i32 6
  %tmp14 = bitcast %struct.__block_byref_repeater* %repeater to i8*
  store i8* %tmp14, i8** %block.captured11, align 8
  %tmp15 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i8* }>* %block to i8*
  %tmp16 = call i8* @objc_retainBlock(i8* %tmp15) nounwind
  %tmp17 = bitcast i8* %tmp16 to void ()*
  %tmp18 = load %struct.__block_byref_repeater** %byref.forwarding, align 8
  %repeater12 = getelementptr inbounds %struct.__block_byref_repeater* %tmp18, i64 0, i32 6
  %tmp13 = load void ()** %repeater12, align 8
  store void ()* %tmp17, void ()** %repeater12, align 8
  ret void
}

declare i8* @objc_retainBlock(i8*)

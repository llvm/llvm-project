; RUN: opt < %s -S -nvptx-lower-args | FileCheck %s --check-prefix IR
; RUN: llc < %s -mcpu=sm_20 | FileCheck %s --check-prefix PTX

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%class.outer = type <{ %class.inner, i32, [4 x i8] }>
%class.inner = type { i32*, i32* }

; Check that nvptx-lower-args preserves arg alignment
define void @load_alignment(%class.outer* nocapture readonly byval(%class.outer) align 8 %arg) {
entry:
; IR: load %class.outer, %class.outer addrspace(101)*
; IR-SAME: align 8
; PTX: ld.param.u64
; PTX-NOT: ld.param.u8
  %arg.idx = getelementptr %class.outer, %class.outer* %arg, i64 0, i32 0, i32 0
  %arg.idx.val = load i32*, i32** %arg.idx, align 8
  %arg.idx1 = getelementptr %class.outer, %class.outer* %arg, i64 0, i32 0, i32 1
  %arg.idx1.val = load i32*, i32** %arg.idx1, align 8
  %arg.idx2 = getelementptr %class.outer, %class.outer* %arg, i64 0, i32 1
  %arg.idx2.val = load i32, i32* %arg.idx2, align 8
  %arg.idx.val.val = load i32, i32* %arg.idx.val, align 4
  %add.i = add nsw i32 %arg.idx.val.val, %arg.idx2.val
  store i32 %add.i, i32* %arg.idx1.val, align 4

  ; let the pointer escape so we still create a local copy this test uses to
  ; check the load alignment.
  %tmp = call i32* @escape(i32* nonnull %arg.idx2)
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local i32* @escape(i32*) local_unnamed_addr

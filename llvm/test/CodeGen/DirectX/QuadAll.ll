; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-library %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-compute %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-pixel %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-amplification %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-mesh %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.0-library %s \
; RUN:   | FileCheck %s --check-prefix=EMULATED --implicit-check-not=quadVote
; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-compute %s \
; RUN:   | FileCheck %s --check-prefix=EMULATED --implicit-check-not=quadVote
; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.6-pixel %s \
; RUN:   | FileCheck %s --check-prefix=EMULATED --implicit-check-not=quadVote
; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.7-library %s | FileCheck %s

; CHECK-LABEL: define i1 @quad_all(
; CHECK: %[[RET:.*]] = call i1 @dx.op.quadVote.i1(i32 222, i1 %cond, i8 1)
; CHECK-NEXT: ret i1 %[[RET]]
; EMULATED-LABEL: define i1 @quad_all(
; EMULATED-NEXT: %[[COND:.*]] = sext i1 %cond to i32
; EMULATED-NEXT: %[[X:.*]] = call i32 @dx.op.quadOp.i32(i32 123, i32 %[[COND]], i8 0)
; EMULATED-NEXT: %[[ANDX:.*]] = and i32 %[[COND]], %[[X]]
; EMULATED-NEXT: %[[Y:.*]] = call i32 @dx.op.quadOp.i32(i32 123, i32 %[[COND]], i8 1)
; EMULATED-NEXT: %[[ANDY:.*]] = and i32 %[[ANDX]], %[[Y]]
; EMULATED-NEXT: %[[D:.*]] = call i32 @dx.op.quadOp.i32(i32 123, i32 %[[COND]], i8 2)
; EMULATED-NEXT: %[[ANDD:.*]] = and i32 %[[ANDY]], %[[D]]
; EMULATED-NEXT: %[[RET:.*]] = trunc i32 %[[ANDD]] to i1
; EMULATED-NEXT: ret i1 %[[RET]]
define i1 @quad_all(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

declare i1 @llvm.dx.quad.all(i1)

; RUN: opt %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; BasicAA should prove that loads from sufficiently large static offsets
; don't overlap with matrix loads with a statically known size.

define <8 x double> @non_overlapping_strided_load(ptr %src) {
entry:
  %src.offset = getelementptr inbounds double, ptr %src, i32 12
  %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
  call void @llvm.matrix.column.major.store(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)
  %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
  %s = fadd <8 x double> %l, %l.2
  ret <8 x double> %s
}

; CHECK-LABEL: Function: non_overlapping_strided_load:
; CHECK: Just Ref: %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)
; CHECK: NoModRef: %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Mod: call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2) <-> %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Mod: call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2) <-> %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: NoModRef: %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Ref: %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)

define <8 x double> @overlapping_strided_load(ptr %src) {
entry:
  %src.offset = getelementptr inbounds double, ptr %src, i32 11
  %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
  call void @llvm.matrix.column.major.store(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)
  %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
  %s = fadd <8 x double> %l, %l.2
  ret <8 x double> %s
}

; CHECK-LABEL: Function: overlapping_strided_load:
; CHECK: Just Ref: %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)
; CHECK: NoModRef: %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Mod: call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2) <-> %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Mod: call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2) <-> %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: NoModRef: %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> %l = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2)
; CHECK: Just Ref: %l.2 = call <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr %src.offset, i32 8, i1 false, i32 4, i32 2) <-> call void @llvm.matrix.column.major.store.v8f64.i32(<8 x double> %l, ptr %src, i32 8, i1 false, i32 4, i32 2)

declare <8 x double> @llvm.matrix.column.major.load.v8f64.i32(ptr, i32, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v8f64.i32(<8 x double>, ptr, i32, i1, i32, i32)

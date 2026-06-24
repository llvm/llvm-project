; RUN: mlir-translate -import-llvm -import-structs-as-literals -split-input-file %s | FileCheck %s

%named = type {i32, i8, i16, i32}

; CHECK: @named
; CHECK-SAME: !llvm.struct<(i32, i8, i16, i32)>
@named = external global %named

%opaque = type opaque

; CHECK: @opaque
; CHECK-SAME: !llvm.struct<()>
@opaque = external global %opaque

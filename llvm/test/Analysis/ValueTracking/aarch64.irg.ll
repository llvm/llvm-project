; RUN: opt -S -instsimplify -instcombine < %s | FileCheck %s

; CHECK-LABEL: define void @checkNonnullIrg()
define void @checkNonnullIrg() {
; CHECK:   %[[p:.*]] = call i8* @llvm.aarch64.irg(i8* nonnull
; CHECK:   call void @use(i8* nonnull %[[p]])
entry:
  %0 = alloca i8, align 16

  %p = call i8* @llvm.aarch64.irg(i8* %0, i64 5)
  call void @use(i8* %p)

  ret void
}

declare i8* @llvm.aarch64.irg(i8*, i64)

declare void @use(i8*)

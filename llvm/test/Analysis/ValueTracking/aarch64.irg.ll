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

; CHECK-LABEL: define void @checkNonnullTagp(
define void @checkNonnullTagp(i8* %tag) {
; CHECK:  %[[p:.*]] = call i8* @llvm.aarch64.tagp.p0i8(i8* nonnull %a, i8* %tag, i64 1)
; CHECK:  %[[p2:.*]] = call i8* @llvm.aarch64.tagp.p0i8(i8* nonnull %[[p]], i8* %tag, i64 2)
; CHECK:  call void @use(i8* nonnull %[[p2]])
entry:
  %a = alloca i8, align 8

  %p = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %tag, i64 1)
  %p2 = call i8* @llvm.aarch64.tagp.p0i8(i8* %p, i8* %tag, i64 2)
  call void @use(i8* %p2)

  ret void
}

declare i8* @llvm.aarch64.irg(i8*, i64)
declare i8* @llvm.aarch64.tagp.p0i8(i8*, i8*, i64)

declare void @use(i8*)

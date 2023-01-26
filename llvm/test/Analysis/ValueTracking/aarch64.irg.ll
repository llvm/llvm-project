; RUN: opt -S -passes=instsimplify,instcombine < %s | FileCheck %s

; CHECK-LABEL: define void @checkNonnullIrg()
define void @checkNonnullIrg() {
; CHECK:   %[[p:.*]] = call ptr @llvm.aarch64.irg(ptr nonnull
; CHECK:   call void @use(ptr nonnull %[[p]])
entry:
  %0 = alloca i8, align 16

  %p = call ptr @llvm.aarch64.irg(ptr %0, i64 5)
  call void @use(ptr %p)

  ret void
}

; CHECK-LABEL: define void @checkNonnullTagp(
define void @checkNonnullTagp(ptr %tag) {
; CHECK:  %[[p:.*]] = call ptr @llvm.aarch64.tagp.p0(ptr nonnull %a, ptr %tag, i64 1)
; CHECK:  %[[p2:.*]] = call ptr @llvm.aarch64.tagp.p0(ptr nonnull %[[p]], ptr %tag, i64 2)
; CHECK:  call void @use(ptr nonnull %[[p2]])
entry:
  %a = alloca i8, align 8

  %p = call ptr @llvm.aarch64.tagp.p0(ptr %a, ptr %tag, i64 1)
  %p2 = call ptr @llvm.aarch64.tagp.p0(ptr %p, ptr %tag, i64 2)
  call void @use(ptr %p2)

  ret void
}

declare ptr @llvm.aarch64.irg(ptr, i64)
declare ptr @llvm.aarch64.tagp.p0(ptr, ptr, i64)

declare void @use(ptr)

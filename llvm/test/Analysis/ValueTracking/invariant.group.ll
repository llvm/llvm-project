; RUN: opt -S -passes=instsimplify,instcombine < %s | FileCheck %s

; CHECK-LABEL: define void @checkNonnullLaunder()
define void @checkNonnullLaunder() {
; CHECK:   %[[p:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %0)
; CHECK:   call void @use(ptr nonnull %[[p]])
entry:
  %0 = alloca i8, align 8

  %p = call ptr @llvm.launder.invariant.group.p0(ptr %0)
  %p2 = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  call void @use(ptr %p2)

  ret void
}

; CHECK-LABEL: define void @checkNonnullStrip()
define void @checkNonnullStrip() {
; CHECK:   %[[p:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr nonnull %0)
; CHECK:   call void @use(ptr nonnull %[[p]])
entry:
  %0 = alloca i8, align 8

  %p = call ptr @llvm.strip.invariant.group.p0(ptr %0)
  %p2 = call ptr @llvm.strip.invariant.group.p0(ptr %p)
  call void @use(ptr %p2)

  ret void
}

declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)

declare void @use(ptr)

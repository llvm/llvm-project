; RUN: opt -S -instsimplify -instcombine < %s | FileCheck %s

; CHECK-LABEL: define void @checkNonnullLaunder()
define void @checkNonnullLaunder() {
; CHECK:   %[[p:.*]] = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %0)
; CHECK:   call void @use(i8* nonnull %[[p]])
entry:
  %0 = alloca i8, align 8

  %p = call i8* @llvm.launder.invariant.group.p0i8(i8* %0)
  %p2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  call void @use(i8* %p2)

  ret void
}

; CHECK-LABEL: define void @checkNonnullStrip()
define void @checkNonnullStrip() {
; CHECK:   %[[p:.*]] = call i8* @llvm.strip.invariant.group.p0i8(i8* nonnull %0)
; CHECK:   call void @use(i8* nonnull %[[p]])
entry:
  %0 = alloca i8, align 8

  %p = call i8* @llvm.strip.invariant.group.p0i8(i8* %0)
  %p2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %p)
  call void @use(i8* %p2)

  ret void
}

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)

declare void @use(i8*)

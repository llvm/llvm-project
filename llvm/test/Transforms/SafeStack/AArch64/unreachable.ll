; RUN: opt -safe-stack -safe-stack-coloring -S -mtriple=aarch64-linux-android < %s -o - | FileCheck %s

define void @foo() nounwind uwtable safestack {
entry:
; CHECK: %[[TP:.*]] = call ptr @llvm.thread.pointer()
; CHECK: %[[SPA0:.*]] = getelementptr i8, ptr %[[TP]], i32 72
; CHECK: %[[USP:.*]] = load ptr, ptr %[[SPA0]]
; CHECK: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; CHECK: store ptr %[[USST]], ptr %[[SPA0]]

  %a = alloca i8, align 8
  br label %ret

ret:
  ret void

dead:
  call void @Capture(ptr %a)
  br label %ret
}

declare void @Capture(ptr)

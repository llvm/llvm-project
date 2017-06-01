; RUN: opt -safe-stack -S -mtriple=aarch64-linux-android < %s -o - | FileCheck %s


define void @foo() nounwind uwtable safestack {
entry:
; CHECK: %[[TP:.*]] = call i8* @llvm.thread.pointer()
; CHECK: %[[SPA0:.*]] = getelementptr i8, i8* %[[TP]], i32 72
; CHECK: %[[SPA:.*]] = bitcast i8* %[[SPA0]] to i8**
; CHECK: %[[USP:.*]] = load i8*, i8** %[[SPA]]
; CHECK: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; CHECK: store i8* %[[USST]], i8** %[[SPA]]

  %a = alloca i8, align 8
  call void @Capture(i8* %a)

; CHECK: store i8* %[[USP]], i8** %[[SPA]]
  ret void
}

declare void @Capture(i8*)

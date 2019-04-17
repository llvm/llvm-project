; RUN: opt -safe-stack -S -mtriple=arm-linux-android < %s -o - | FileCheck %s


define void @foo() nounwind uwtable safestack {
entry:
; CHECK: %[[SPA:.*]] = call i8** @__safestack_pointer_address()
; CHECK: %[[USP:.*]] = load i8*, i8** %[[SPA]]
; CHECK: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; CHECK: store i8* %[[USST]], i8** %[[SPA]]

  %a = alloca i8, align 8
  call void @Capture(i8* %a)

; CHECK: store i8* %[[USP]], i8** %[[SPA]]
  ret void
}

declare void @Capture(i8*)

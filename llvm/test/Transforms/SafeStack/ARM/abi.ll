; RUN: opt -safe-stack -S -mtriple=arm-linux-android < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=arm-linux-android < %s -o - | FileCheck %s


define void @foo() nounwind uwtable safestack {
entry:
; CHECK: %[[SPA:.*]] = call ptr @__safestack_pointer_address()
; CHECK: %[[USP:.*]] = load ptr, ptr %[[SPA]]
; CHECK: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; CHECK: store ptr %[[USST]], ptr %[[SPA]]

  %a = alloca i8, align 8
  call void @Capture(ptr %a)

; CHECK: store ptr %[[USP]], ptr %[[SPA]]
  ret void
}

declare void @Capture(ptr)

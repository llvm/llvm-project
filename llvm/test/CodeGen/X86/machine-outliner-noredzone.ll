; RUN: llc -enable-machine-outliner -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; Ensure that the outliner doesn't outline from any functions that use a redzone.

declare ptr @llvm.stacksave() #1
declare void @llvm.stackrestore(ptr) #1

; This function has a red zone. We shouldn't outline from it.
; CHECK-LABEL: doggo
; CHECK-NOT: OUTLINED
define void @doggo(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 4
  ret void
}

; Ditto.
; CHECK-LABEL: pupper
; CHECK-NOT: OUTLINED
define void @pupper(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 4
  ret void
}

; This doesn't have a redzone. Outlining is okay.
; CHECK-LABEL: boofer
; CHECK: OUTLINED
define void @boofer(i32) #0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store i32 %0, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = zext i32 %5 to i64
  %7 = call ptr @llvm.stacksave()
  store ptr %7, ptr %3, align 8
  %8 = alloca i32, i64 %6, align 16
  store i64 %6, ptr %4, align 8
  %9 = load ptr, ptr %3, align 8
  call void @llvm.stackrestore(ptr %9)
  ret void
}

; Ditto.
; CHECK-LABEL: shibe
; CHECK: OUTLINED
define void @shibe(i32) #0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store i32 %0, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = zext i32 %5 to i64
  %7 = call ptr @llvm.stacksave()
  store ptr %7, ptr %3, align 8
  %8 = alloca i32, i64 %6, align 16
  store i64 %6, ptr %4, align 8
  %9 = load ptr, ptr %3, align 8
  call void @llvm.stackrestore(ptr %9)
  ret void
}

attributes #0 = { noinline nounwind optnone ssp uwtable "frame-pointer"="all" }
attributes #1 = { nounwind }
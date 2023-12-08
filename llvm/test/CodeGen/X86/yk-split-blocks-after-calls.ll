; RUN: llc -stop-after yk-splitblocksaftercalls --yk-split-blocks-after-calls --yk-no-calls-in-entryblocks < %s  | FileCheck %s

; Check that the block in the main function containing two calls to foo, has
; been split after each of the calls.

; CHECK-LABEL: define dso_local i32 @main
; CHECK-NEXT: %2 = add nsw i32 %0, 1
; CHECK-NEXT: br label %3
; CHECK-LABEL: 3:
; CHECK-NEXT: %4 = call i32 @foo(i32 noundef %2)
; CHECK-NEXT: br label %5
; CHECK-LABEL: 5:
; CHECK-NEXT: %6 = add nsw i32 %0, 1
; CHECK-NEXT: %7 = call i32 @foo(i32 noundef %4)
; CHECK-NEXT: br label %8

@.str = private unnamed_addr constant [13 x i8] c"%d %d %d %d\0A\00", align 1

define dso_local i32 @foo(i32 noundef %0) #0 {
  %2 = add nsw i32 %0, 10
  ret i32 %2
}

declare i32 @printf(ptr noundef, ...) #2

define dso_local i32 @main(i32 noundef %0) #0 {
  %2 = add nsw i32 %0, 1
  %3 = call i32 @foo(i32 noundef %2)
  %4 = add nsw i32 %0, 1
  %5 = call i32 @foo(i32 noundef %3)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

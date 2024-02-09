; RUN: llc -stop-after=yk-stackmaps --yk-insert-stackmaps < %s  | FileCheck %s

; Check whether `llvm.experimental.stackmap` calls are only inserted after normal calls,
; and not after external calls.

; CHECK-LABEL: define dso_local i32 @main
; CHECK: %call = call i32 @foo(i32 noundef %0)
; CHECK-NEXT: call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, ptr %argc.addr, ptr %a, ptr %b, i32 %0)
; CHECK: %call1 = call i32 @foo(i32 noundef %1)
; CHECK-NEXT: call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, ptr %argc.addr, ptr %a, ptr %b, i32 %1)
; CHECK: %call2 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %2, i32 noundef %3, i32 noundef %4)
; CHECK-NOT: call void (i64, i32, ...) @llvm.experimental.stackmap

@.str = private unnamed_addr constant [10 x i8] c"%d %d %d\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @foo(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %add = add nsw i32 %0, 10
  ret i32 %add
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  %0 = load i32, ptr %argc.addr, align 4
  %call = call i32 @foo(i32 noundef %0)
  store i32 %call, ptr %a, align 4
  %1 = load i32, ptr %a, align 4
  %call1 = call i32 @foo(i32 noundef %1)
  store i32 %call1, ptr %b, align 4
  %2 = load i32, ptr %argc.addr, align 4
  %3 = load i32, ptr %a, align 4
  %4 = load i32, ptr %b, align 4
  %call2 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %2, i32 noundef %3, i32 noundef %4)
  ret i32 0
}

declare dso_local i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

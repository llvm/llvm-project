; RUN: llc -mtriple=aarch64-apple-darwin < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; CHECK-LABEL: test1:
; CHECK-NOT: ___stack_chk_guard

; Function Attrs: noinline nounwind optnone
define void @test1(ptr noundef %msg) #0 {
entry:
  %msg.addr = alloca ptr, align 8
  %a = alloca [1000 x i8], align 1, !stack-protector !2
  store ptr %msg, ptr %msg.addr, align 8
  %arraydecay = getelementptr inbounds [1000 x i8], ptr %a, i64 0, i64 0
  %0 = load ptr, ptr %msg.addr, align 8
  %call = call ptr @strcpy(ptr noundef %arraydecay, ptr noundef %0) #3
  %arraydecay1 = getelementptr inbounds [1000 x i8], ptr %a, i64 0, i64 0
  %call2 = call i32 (ptr, ...) @printf(ptr noundef @.str, ptr noundef %arraydecay1)
  ret void
}


; CHECK-LABEL: test2:
; CHECK: ___stack_chk_guard

; Function Attrs: noinline nounwind optnone
define void @test2(ptr noundef %msg) #0 {
entry:
  %msg.addr = alloca ptr, align 8
  %b = alloca [1000 x i8], align 1
  store ptr %msg, ptr %msg.addr, align 8
  %arraydecay = getelementptr inbounds [1000 x i8], ptr %b, i64 0, i64 0
  %0 = load ptr, ptr %msg.addr, align 8
  %call = call ptr @strcpy(ptr noundef %arraydecay, ptr noundef %0) #3
  %arraydecay1 = getelementptr inbounds [1000 x i8], ptr %b, i64 0, i64 0
  %call2 = call i32 (ptr, ...) @printf(ptr noundef @.str, ptr noundef %arraydecay1)
  ret void
}

; Function Attrs: nounwind
declare ptr @strcpy(ptr noundef, ptr noundef) #1

declare i32 @printf(ptr noundef, ...) #2

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" ssp }
attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 22.0.0"}
!2 = !{i32 0}

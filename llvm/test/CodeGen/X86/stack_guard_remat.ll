; RUN: llc < %s -mtriple=x86_64-apple-darwin -no-integrated-as | FileCheck %s

;CHECK:  foo2
;CHECK:  movq ___stack_chk_guard@GOTPCREL(%rip), [[R0:%[a-z0-9]+]]
;CHECK:  movq ([[R0]]), {{%[a-z0-9]+}}

; Function Attrs: nounwind ssp uwtable
define i32 @test_stack_guard_remat() #0 {
entry:
  %a1 = alloca [256 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 1024, ptr %a1)
  call void @foo3(ptr %a1)
  call void asm sideeffect "foo2", "~{r12},~{r13},~{r14},~{r15},~{ebx},~{esi},~{edi},~{dirflag},~{fpsr},~{flags}"()
  call void @llvm.lifetime.end.p0(i64 1024, ptr %a1)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @foo3(ptr)

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

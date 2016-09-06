; RUN: llc < %s -mtriple=x86_64-apple-darwin -no-integrated-as | FileCheck %s

;CHECK:  foo2
;CHECK:  movq ___stack_chk_guard@GOTPCREL(%rip), [[R0:%[a-z0-9]+]]
;CHECK:  movq ([[R0]]), {{%[a-z0-9]+}}

; Function Attrs: nounwind ssp uwtable
define i32 @test_stack_guard_remat() #0 {
entry:
  %a1 = alloca [256 x i32], align 16
  %0 = bitcast [256 x i32]* %a1 to i8*
  call void @llvm.lifetime.start(i64 1024, i8* %0)
  %arraydecay = getelementptr inbounds [256 x i32], [256 x i32]* %a1, i64 0, i64 0
  call void @foo3(i32* %arraydecay)
  call void asm sideeffect "foo2", "~{r12},~{r13},~{r14},~{r15},~{ebx},~{esi},~{edi},~{dirflag},~{fpsr},~{flags}"()
  call void @llvm.lifetime.end(i64 1024, i8* %0)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @foo3(i32*)

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture)

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

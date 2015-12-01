; RUN: llc -mtriple=x86_64-apple-darwin -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=true %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SELDAG
; RUN: llc -mtriple=x86_64-apple-darwin -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=false %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=IR

; SELDAG: # Machine code for function test_branch_weights:
; SELDAG: Successors according to CFG: BB#[[SUCCESS:[0-9]+]]({{[0-9a-fx/= ]+}}100.00%) BB#[[FAILURE:[0-9]+]]
; SELDAG: BB#[[FAILURE]]:
; SELDAG: CALL64pcrel32 <es:__stack_chk_fail>
; SELDAG: BB#[[SUCCESS]]:

; IR: # Machine code for function test_branch_weights:
; IR: Successors according to CFG: BB#[[SUCCESS:[0-9]+]]({{[0-9a-fx/= ]+}}100.00%) BB#[[FAILURE:[0-9]+]]
; IR: BB#[[SUCCESS]]:
; IR: BB#[[FAILURE]]:
; IR: CALL64pcrel32 <ga:@__stack_chk_fail>

define i32 @test_branch_weights(i32 %n) #0 {
entry:
  %a = alloca [128 x i32], align 16
  %0 = bitcast [128 x i32]* %a to i8*
  call void @llvm.lifetime.start(i64 512, i8* %0)
  %arraydecay = getelementptr inbounds [128 x i32], [128 x i32]* %a, i64 0, i64 0
  call void @foo2(i32* %arraydecay)
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32]* %a, i64 0, i64 %idxprom
  %1 = load i32, i32* %arrayidx, align 4
  call void @llvm.lifetime.end(i64 512, i8* %0)
  ret i32 %1
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @foo2(i32*)

declare void @llvm.lifetime.end(i64, i8* nocapture)

attributes #0 = { ssp "stack-protector-buffer-size"="8" }

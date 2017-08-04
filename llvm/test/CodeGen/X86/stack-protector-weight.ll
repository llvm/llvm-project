; RUN: llc -mtriple=x86_64-apple-darwin -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=true %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DARWIN-SELDAG
; RUN: llc -mtriple=x86_64-apple-darwin -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=false %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DARWIN-IR
; RUN: llc -mtriple=i386-pc-windows-msvc -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=true %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=MSVC-SELDAG
; RUN: llc -mtriple=i386-pc-windows-msvc -print-machineinstrs=expand-isel-pseudos -enable-selectiondag-sp=false %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=MSVC-IR

; DARWIN-SELDAG: # Machine code for function test_branch_weights:
; DARWIN-SELDAG: Successors according to CFG: BB#[[SUCCESS:[0-9]+]]({{[0-9a-fx/= ]+}}100.00%) BB#[[FAILURE:[0-9]+]]
; DARWIN-SELDAG: BB#[[FAILURE]]:
; DARWIN-SELDAG: CALL64pcrel32 <es:__stack_chk_fail>
; DARWIN-SELDAG: BB#[[SUCCESS]]:

; DARWIN-IR: # Machine code for function test_branch_weights:
; DARWIN-IR: Successors according to CFG: BB#[[SUCCESS:[0-9]+]]({{[0-9a-fx/= ]+}}100.00%) BB#[[FAILURE:[0-9]+]]
; DARWIN-IR: BB#[[SUCCESS]]:
; DARWIN-IR: BB#[[FAILURE]]:
; DARWIN-IR: CALL64pcrel32 <ga:@__stack_chk_fail>

; MSVC-SELDAG: # Machine code for function test_branch_weights:
; MSVC-SELDAG: mem:Volatile LD4[@__security_cookie]
; MSVC-SELDAG: ST4[FixedStack0]
; MSVC-SELDAG: LD4[FixedStack0]
; MSVC-SELDAG: CALLpcrel32 <ga:@__security_check_cookie>

; MSVC-IR: # Machine code for function test_branch_weights:
; MSVC-IR: mem:Volatile LD4[@__security_cookie]
; MSVC-IR: ST4[FixedStack0]
; MSVC-IR: LD4[%StackGuardSlot]
; MSVC-IR: CALLpcrel32 <ga:@__security_check_cookie>

define i32 @test_branch_weights(i32 %n) #0 {
entry:
  %a = alloca [128 x i32], align 16
  %0 = bitcast [128 x i32]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 512, i8* %0)
  %arraydecay = getelementptr inbounds [128 x i32], [128 x i32]* %a, i64 0, i64 0
  call void @foo2(i32* %arraydecay)
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32]* %a, i64 0, i64 %idxprom
  %1 = load i32, i32* %arrayidx, align 4
  call void @llvm.lifetime.end.p0i8(i64 512, i8* %0)
  ret i32 %1
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @foo2(i32*)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

attributes #0 = { sspstrong "stack-protector-buffer-size"="8" }

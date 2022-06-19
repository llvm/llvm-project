; RUN: llc -max-registers-for-gc-values=4 -stop-after virtregrewriter < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare dso_local void @"some_call"(i64 addrspace(1)*)
declare dso_local i32 @foo(i32, i8 addrspace(1)*, i32, i32, i32)
declare dso_local i32* @personality_function()

define i64 addrspace(1)* @test_basic_invoke(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
; CHECK-LABEL:    name: test_basic_invoke
; CHECK:          bb.0.entry:
; CHECK:          MOV64mr %stack.1, 1, $noreg, 0, $noreg, renamable $rdi :: (store (s64) into %stack.1)
; CHECK:          MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rsi :: (store (s64) into %stack.0)
; CHECK:          STATEPOINT 0, 0, 1, @some_call, $rdi, 2, 0, 2, 0, 2, 5, 2, 0, 2, -1, 2, 0, 2, 0, 2, 0, 2, 2, 1, 8, %stack.0, 0, 1, 8, %stack.1, 0, 2, 0, 2, 2, 0, 0, 1, 1, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store (s64) on %stack.0), (volatile load store (s64) on %stack.1)
; CHECK:          JMP_1 %bb.1
; CHECK:          bb.1.safepoint_normal_dest:
; CHECK:          renamable $rax = MOV64rm %stack.1, 1, $noreg, 0, $noreg :: (load (s64) from %stack.1)
; CHECK:          bb.2.normal_return:
; CHECK:          RET 0, $rax
; CHECK:          bb.3.exceptional_return (landing-pad):
; CHECK:          renamable $rax = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          RET 0, $rax
  gc "statepoint-example" personality i32* ()* @"personality_function" {
entry:
  %0 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* elementtype(void (i64 addrspace(1)*)) @some_call, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 0) ["gc-live" (i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1), "deopt" (i32 0, i32 -1, i32 0, i32 0, i32 0)]
          to label %safepoint_normal_dest unwind label %exceptional_return

safepoint_normal_dest:
  %obj.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %0, i32 0, i32 0)
  %obj1.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %0, i32 1, i32 1)
  br label %normal_return

normal_return:
  ret i64 addrspace(1)* %obj.relocated

exceptional_return:
  %landing_pad = landingpad token
          cleanup
  %obj.relocated1 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 0, i32 0)
  %obj1.relocated1 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 1, i32 1)
  ret i64 addrspace(1)* %obj1.relocated1
}

define i64 addrspace(1)* @test_invoke_same_val(i1 %cond, i64 addrspace(1)* %val1, i64 addrspace(1)* %val2, i64 addrspace(1)* %val3)
; CHECK-LABEL:    name: test_invoke_same_val
; CHECK:          bb.0.entry:
; CHECK:          renamable $rbx = COPY $rcx
; CHECK:          renamable $rbp = COPY $rdx
; CHECK:          renamable $r14d = COPY $edi
; CHECK:          TEST8ri renamable $r14b, 1, implicit-def $eflags
; CHECK:          JCC_1 %bb.3, 4, implicit killed $eflags
; CHECK:          JMP_1 %bb.1
; CHECK:          bb.1.left:
; CHECK:          MOV64mr %stack.0, 1, $noreg, 0, $noreg, renamable $rsi :: (store (s64) into %stack.0)
; CHECK:          $rdi = COPY killed renamable $rsi
; CHECK:          renamable $rbp = STATEPOINT 0, 0, 1, @some_call, $rdi, 2, 0, 2, 0, 2, 0, 2, 2, killed renamable $rbp(tied-def 0), 1, 8, %stack.0, 0, 2, 0, 2, 2, 0, 0, 1, 1, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store (s64) on %stack.0)
; CHECK:          JMP_1 %bb.2
; CHECK:          bb.2.left.relocs:
; CHECK:          renamable $rbx = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          JMP_1 %bb.5
; CHECK:          bb.3.right:
; CHECK:          MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rbp :: (store (s64) into %stack.0)
; CHECK:          $rdi = COPY killed renamable $rsi
; CHECK:          renamable $rbx = STATEPOINT 0, 0, 1, @some_call, $rdi, 2, 0, 2, 0, 2, 0, 2, 2, killed renamable $rbx(tied-def 0), 1, 8, %stack.0, 0, 2, 0, 2, 2, 0, 0, 1, 1, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store (s64) on %stack.0)
; CHECK:          JMP_1 %bb.4
; CHECK:          bb.4.right.relocs:
; CHECK:          renamable $rbp = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          bb.5.normal_return:
; CHECK:          TEST8ri renamable $r14b, 1, implicit-def $eflags, implicit killed $r14d
; CHECK:          renamable $rbx = CMOV64rr killed renamable $rbx, killed renamable $rbp, 4, implicit killed $eflags
; CHECK:          $rax = COPY killed renamable $rbx
; CHECK:          RET 0, $rax
; CHECK:          bb.6.exceptional_return.left (landing-pad):
; CHECK:          renamable $rax = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          RET 0, $rax
; CHECK:          bb.7.exceptional_return.right (landing-pad):
; CHECK:          renamable $rax = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          RET 0, $rax
  gc "statepoint-example" personality i32* ()* @"personality_function" {
entry:
  br i1 %cond, label %left, label %right

left:
  %sp1 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* elementtype(void (i64 addrspace(1)*)) @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 0) ["gc-live"(i64 addrspace(1)* %val1, i64 addrspace(1)* %val2)]
           to label %left.relocs unwind label %exceptional_return.left

left.relocs:
  %val1.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 0, i32 0)
  %val2.relocated_left = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp1, i32 1, i32 1)
  br label %normal_return

right:
  %sp2 = invoke token (i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64 0, i32 0, void (i64 addrspace(1)*)* elementtype(void (i64 addrspace(1)*)) @some_call, i32 1, i32 0, i64 addrspace(1)* %val1, i32 0, i32 0) ["gc-live"(i64 addrspace(1)* %val2, i64 addrspace(1)* %val3)]
           to label %right.relocs unwind label %exceptional_return.right

right.relocs:
  %val2.relocated_right = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp2, i32 0, i32 0)
  %val3.relocated = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %sp2, i32 1, i32 1)
  br label %normal_return

normal_return:
  %a1 = phi i64 addrspace(1)* [%val1.relocated, %left.relocs], [%val3.relocated, %right.relocs]
  %a2 = phi i64 addrspace(1)* [%val2.relocated_left, %left.relocs], [%val2.relocated_right, %right.relocs]
  %ret = select i1 %cond, i64 addrspace(1)* %a1, i64 addrspace(1)* %a2
  ret i64 addrspace(1)* %ret

exceptional_return.left:
  %landing_pad = landingpad token
          cleanup
  %val1.relocated2 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad, i32 0, i32 0)
  ret i64 addrspace(1)* %val1.relocated2

exceptional_return.right:
  %landing_pad1 = landingpad token
          cleanup
  %val2.relocated3 = call coldcc i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %landing_pad1, i32 0, i32 0)
  ret i64 addrspace(1)* %val2.relocated3
}

define void @test_duplicate_ir_values() gc "statepoint-example" personality i32* ()* @personality_function {
; CHECK-LABEL:    name: test_duplicate_ir_values
; CHECK:          bb.0.entry:
; CHECK:          renamable $rax = MOV64rm undef renamable $rax, 1, $noreg, 0, $noreg :: (load (s64) from `i8 addrspace(1)* addrspace(1)* undef`, addrspace 1)
; CHECK:          MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rax :: (store (s64) into %stack.0)
; CHECK:          STATEPOINT 1, 16, 5, undef renamable $rax, undef $edi, undef $rsi, undef $edx, undef $ecx, undef $r8d, 2, 0, 2, 0, 2, 0, 2, 1, 1, 8, %stack.0, 0, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def dead $eax :: (volatile load store (s64) on %stack.0)
; CHECK:          JMP_1 %bb.1
; CHECK:          bb.1.normal_continue:
; CHECK:          renamable $rbx = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          $edi = MOV32ri 10
; CHECK:          dead renamable $rbx = STATEPOINT 2882400000, 0, 1, target-flags(x86-plt) @__llvm_deoptimize, killed $edi, 2, 0, 2, 2, 2, 2, killed renamable $rbx, renamable $rbx, 2, 1, renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK:          bb.2.exceptional_return (landing-pad):
; CHECK:          renamable $rbx = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load (s64) from %stack.0)
; CHECK:          $edi = MOV32ri -271
; CHECK:          dead renamable $rbx = STATEPOINT 2882400000, 0, 1, target-flags(x86-plt) @__llvm_deoptimize, killed $edi, 2, 0, 2, 0, 2, 1, killed renamable $rbx, 2, 1, renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
entry:
  %val1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* undef, align 8
  %val2 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* undef, align 8
  %statepoint_token1 = invoke token (i64, i32, i32 (i32, i8 addrspace(1)*, i32, i32, i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32i32i32f(i64 1, i32 16, i32 (i32, i8 addrspace(1)*, i32, i32, i32)* nonnull elementtype(i32 (i32, i8 addrspace(1)*, i32, i32, i32)) @foo, i32 5, i32 0, i32 undef, i8 addrspace(1)* undef, i32 undef, i32 undef, i32 undef, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* %val1, i8 addrspace(1)* %val2) ]
          to label %normal_continue unwind label %exceptional_return

normal_continue: ; preds = %entry
  %val1.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1, i32 0, i32 0) ; (%val1, %val1)
  %val2.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1, i32 1, i32 1) ; (%val2, %val2)
  %safepoint_token2 = call token (i64, i32, void (i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 2882400000, i32 0, void (i32)* nonnull elementtype(void (i32)) @__llvm_deoptimize, i32 1, i32 2, i32 10, i32 0, i32 0) [ "deopt"(i8 addrspace(1)* %val1.relocated1, i8 addrspace(1)* %val2.relocated1), "gc-live"() ]
  unreachable

exceptional_return:                         ; preds = %entry
  %lpad_token11090 = landingpad token
          cleanup
  %val2.relocated2 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %lpad_token11090, i32 1, i32 1) ; (%val2, %val2)
  %safepoint_token3 = call token (i64, i32, void (i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 2882400000, i32 0, void (i32)* nonnull elementtype(void (i32)) @__llvm_deoptimize, i32 1, i32 0, i32 -271, i32 0, i32 0) [ "deopt"(i8 addrspace(1)* %val2.relocated2), "gc-live"() ]
  unreachable
}

declare void @__llvm_deoptimize(i32)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 immarg, i32 immarg, void (i32)*, i32 immarg, i32 immarg, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidp1i64f(i64, i32, void (i64 addrspace(1)*)*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32i32i32f(i64 immarg, i32 immarg, i32 (i32, i8 addrspace(1)*, i32, i32, i32)*, i32 immarg, i32 immarg, ...)
declare i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token, i32, i32)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)

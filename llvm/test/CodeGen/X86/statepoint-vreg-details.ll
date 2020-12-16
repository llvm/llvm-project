; This file contains some of the same basic tests as statepoint-vreg.ll, but
; focuses on examining the intermediate representation.  It's separate so that
; the main file is easy to update with update_llc_test_checks.py

; This run is to demonstrate what MIR SSA looks like.
; RUN: llc -max-registers-for-gc-values=4 -stop-after finalize-isel < %s | FileCheck --check-prefix=CHECK-VREG %s
; This run is to demonstrate register allocator work.
; RUN: llc -max-registers-for-gc-values=4 -stop-after virtregrewriter < %s | FileCheck --check-prefix=CHECK-PREG %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare dso_local i1 @return_i1()
declare dso_local void @func()
declare dso_local void @consume(i32 addrspace(1)*)
declare dso_local void @consume2(i32 addrspace(1)*, i32 addrspace(1)*)
declare dso_local void @consume5(i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*)
declare dso_local void @use1(i32 addrspace(1)*, i8 addrspace(1)*)
declare dso_local void @bar(i8 addrspace(1)*, i8 addrspace(1)*)

; test most simple relocate
define i1 @test_relocate(i32 addrspace(1)* %a) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_relocate
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, %0(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al
; CHECK-VREG:    %2:gr8 = COPY $al
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_relocate
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, killed renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al
; CHECK-PREG:    renamable $bpl = COPY killed $al
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %res1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  call void @consume(i32 addrspace(1)* %rel1)
  ret i1 %res1
}
; test pointer variables intermixed with pointer constants
define void @test_mixed(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_mixed
; CHECK-VREG:    %2:gr64 = COPY $rdx
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %3:gr64, %4:gr64, %5:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 4, %2(tied-def 0), 2, 0, %1(tied-def 1), %0(tied-def 2), 2, 0, 2, 4, 0, 0, 1, 1, 2, 2, 3, 3, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    %6:gr32 = MOV32r0 implicit-def dead $eflags
; CHECK-VREG:    %7:gr64 = SUBREG_TO_REG 0, killed %6, %subreg.sub_32bit
; CHECK-VREG:    $rdi = COPY %5
; CHECK-VREG:    $rsi = COPY %7
; CHECK-VREG:    $rdx = COPY %4
; CHECK-VREG:    $rcx = COPY %7
; CHECK-VREG:    $r8 = COPY %3
; CHECK-VREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_mixed
; CHECK-PREG:    renamable $r14 = COPY $rdx
; CHECK-PREG:    renamable $r15 = COPY $rsi
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $r14, renamable $r15, renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 4, killed renamable $r14(tied-def 0), 2, 0, killed renamable $r15(tied-def 1), killed renamable $rbx(tied-def 2), 2, 0, 2, 4, 0, 0, 1, 1, 2, 2, 3, 3, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    dead $esi = MOV32r0 implicit-def dead $eflags, implicit-def $rsi
; CHECK-PREG:    $rdx = COPY killed renamable $r15
; CHECK-PREG:    dead $ecx = MOV32r0 implicit-def dead $eflags, implicit-def $rcx
; CHECK-PREG:    $r8 = COPY killed renamable $r14
; CHECK-PREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit killed $rcx, implicit killed $r8, implicit-def $rsp, implicit-def $ssp

entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a, i32 addrspace(1)* null, i32 addrspace(1)* %b, i32 addrspace(1)* null, i32 addrspace(1)* %c)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  %rel3 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 2, i32 2)
  %rel4 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 3, i32 3)
  %rel5 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 4, i32 4)
  call void @consume5(i32 addrspace(1)* %rel1, i32 addrspace(1)* %rel2, i32 addrspace(1)* %rel3, i32 addrspace(1)* %rel4, i32 addrspace(1)* %rel5)
  ret void
}

; same as above, but for alloca
define i32 addrspace(1)* @test_alloca(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_alloca
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0.alloca, 1, $noreg, 0, $noreg, %0 :: (store 8 into %ir.alloca)
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, %0(tied-def 0), 2, 1, 0, %stack.0.alloca, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al :: (volatile load store 8 on %stack.0.alloca)
; CHECK-VREG:    %2:gr8 = COPY $al
; CHECK-VREG:    %3:gr64 = MOV64rm %stack.0.alloca, 1, $noreg, 0, $noreg :: (dereferenceable load 8 from %ir.alloca)
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_alloca
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    MOV64mr %stack.0.alloca, 1, $noreg, 0, $noreg, renamable $rbx :: (store 8 into %ir.alloca)
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, killed renamable $rbx(tied-def 0), 2, 1, 0, %stack.0.alloca, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def dead $al :: (volatile load store 8 on %stack.0.alloca)
; CHECK-PREG:    renamable $r14 = MOV64rm %stack.0.alloca, 1, $noreg, 0, $noreg :: (dereferenceable load 8 from %ir.alloca)
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

entry:
  %alloca = alloca i32 addrspace(1)*, align 8
  store i32 addrspace(1)* %ptr, i32 addrspace(1)** %alloca
  %safepoint_token = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)** %alloca, i32 addrspace(1)* %ptr)]
  %rel1 = load i32 addrspace(1)*, i32 addrspace(1)** %alloca
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  call void @consume(i32 addrspace(1)* %rel2)
  ret i32 addrspace(1)* %rel1
}

; test base != derived
define void @test_base_derived(i32 addrspace(1)* %base, i32 addrspace(1)* %derived) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_base_derived
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %2:gr64, %3:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 2, %1(tied-def 0), %0(tied-def 1), 2, 0, 2, 1, 1, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    $rdi = COPY %2
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_base_derived
; CHECK-PREG:    renamable $rbx = COPY $rsi
; CHECK-PREG:    renamable $rbx, dead renamable $r14 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 2, killed renamable $rbx(tied-def 0), killed renamable $r14(tied-def 1), 2, 0, 2, 1, 1, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %base, i32 addrspace(1)* %derived)]
  %reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 1)
  call void @consume(i32 addrspace(1)* %reloc)
  ret void
}

; deopt GC pointer not present in GC args must be spilled
define void @test_deopt_gcpointer(i32 addrspace(1)* %a, i32 addrspace(1)* %b) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_deopt_gcpointer
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG:    %2:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 1, 1, 8, %stack.0, 0, 2, 1, %1(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-VREG:    $rdi = COPY %2
; CHECK-VREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    RET 0

; CHECK-PREG-LABEL: name:            test_deopt_gcpointer
; CHECK-PREG:    renamable $rbx = COPY $rsi
; CHECK-PREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, killed renamable $rdi :: (store 8 into %stack.0)
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 1, 1, 8, %stack.0, 0, 2, 1, killed renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-PREG:    $rdi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp

  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %a), "gc-live" (i32 addrspace(1)* %b)]
  %rel = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  call void @consume(i32 addrspace(1)* %rel)
  ret void
}

;; Two gc.relocates of the same input, should require only a single spill/fill
define void @test_gcrelocate_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_gcrelocate_uniqueing
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, %0, 2, 4278124286, 2, 1, %0(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    $rsi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @consume2, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_gcrelocate_uniqueing
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, killed renamable $rbx, 2, 4278124286, 2, 1, renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY renamable $rbx
; CHECK-PREG:    $rsi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @consume2, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit killed $rsi, implicit-def $rsp, implicit-def $ssp

  %tok = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %ptr, i32 undef), "gc-live" (i32 addrspace(1)* %ptr, i32 addrspace(1)* %ptr)]
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 0, i32 0)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 1, i32 1)
  call void @consume2(i32 addrspace(1)* %a, i32 addrspace(1)* %b)
  ret void
}

; Two gc.relocates of a bitcasted pointer should only require a single spill/fill
define void @test_gcptr_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_gcptr_uniqueing
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    %1:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, %0, 2, 4278124286, 2, 1, %0(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG:    $rdi = COPY %1
; CHECK-VREG:    $rsi = COPY %1
; CHECK-VREG:    CALL64pcrel32 @use1, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp

; CHECK-PREG-LABEL: name:            test_gcptr_uniqueing
; CHECK-PREG:    renamable $rbx = COPY $rdi
; CHECK-PREG:    renamable $rbx = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 2, killed renamable $rbx, 2, 4278124286, 2, 1, renamable $rbx(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-PREG:    $rdi = COPY renamable $rbx
; CHECK-PREG:    $rsi = COPY killed renamable $rbx
; CHECK-PREG:    CALL64pcrel32 @use1, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit killed $rsi, implicit-def $rsp, implicit-def $ssp

  %ptr2 = bitcast i32 addrspace(1)* %ptr to i8 addrspace(1)*
  %tok = tail call token (i64, i32, void ()*, i32, i32, ...)
      @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 addrspace(1)* %ptr, i32 undef), "gc-live" (i32 addrspace(1)* %ptr, i8 addrspace(1)* %ptr2)]
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 0, i32 0)
  %b = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tok, i32 1, i32 1)
  call void @use1(i32 addrspace(1)* %a, i8 addrspace(1)* %b)
  ret void
}

define i1 @test_cross_bb(i32 addrspace(1)* %a, i1 %external_cond) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_cross_bb
; CHECK-VREG:  bb.0.entry:
; CHECK-VREG:         %1:gr32 = COPY $esi
; CHECK-VREG-NEXT:    %0:gr64 = COPY $rdi
; CHECK-VREG-NEXT:    %4:gr8 = COPY %1.sub_8bit
; CHECK-VREG-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    %2:gr64 = STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, %0(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp, implicit-def $al
; CHECK-VREG-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    %5:gr8 = COPY $al
; CHECK-VREG-NEXT:    %3:gr8 = COPY %5
; CHECK-VREG-NEXT:    TEST8ri killed %4, 1, implicit-def $eflags
; CHECK-VREG-NEXT:    JCC_1 %bb.2, 4, implicit $eflags
; CHECK-VREG-NEXT:    JMP_1 %bb.1
; CHECK-VREG:       bb.1.left:
; CHECK-VREG-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    $rdi = COPY %2
; CHECK-VREG-NEXT:    CALL64pcrel32 @consume, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
; CHECK-VREG-NEXT:    $al = COPY %3
; CHECK-VREG-NEXT:    RET 0, $al
; CHECK-VREG:       bb.2.right:
; CHECK-VREG-NEXT:    %6:gr8 = MOV8ri 1
; CHECK-VREG-NEXT:    $al = COPY %6
; CHECK-VREG-NEXT:    RET 0, $al

entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a)]
  br i1 %external_cond, label %left, label %right

left:
  %call1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %call2 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  call void @consume(i32 addrspace(1)* %call1)
  ret i1 %call2

right:
  ret i1 true
}

; No need to check post-regalloc output as it is the same
define i1 @duplicate_reloc() gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            duplicate_reloc
; CHECK-VREG:  bb.0.entry:
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    %0:gr8 = MOV8ri 1
; CHECK-VREG:    $al = COPY %0
; CHECK-VREG:    RET 0, $al

entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* null, i32 addrspace(1)* null)]
  %base = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %derived = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 1)
  %safepoint_token2 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %base, i32 addrspace(1)* %derived)]
  %base_reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2,  i32 0, i32 0)
  %derived_reloc = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2,  i32 0, i32 1)
  %cmp1 = icmp eq i32 addrspace(1)* %base_reloc, null
  %cmp2 = icmp eq i32 addrspace(1)* %derived_reloc, null
  %cmp = and i1 %cmp1, %cmp2
  ret i1 %cmp
}

; Vectors cannot go in VRegs
; No need to check post-regalloc output as it is lowered using old scheme
define <2 x i8 addrspace(1)*> @test_vector(<2 x i8 addrspace(1)*> %obj) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_vector
; CHECK-VREG:    %0:vr128 = COPY $xmm0
; CHECK-VREG:    MOVAPSmr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 16 into %stack.0)
; CHECK-VREG:    STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 1, 1, 16, %stack.0, 0, 2, 0, 2, 1, 0, 0, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 16 on %stack.0)
; CHECK-VREG:    %1:vr128 = MOVAPSrm %stack.0, 1, $noreg, 0, $noreg :: (load 16 from %stack.0)
; CHECK-VREG:    $xmm0 = COPY %1
; CHECK-VREG:    RET 0, $xmm0

entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (<2 x i8 addrspace(1)*> %obj)]
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 0, i32 0) ; (%obj, %obj)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}


; test limit on amount of vregs
define void @test_limit(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c, i32 addrspace(1)* %d, i32 addrspace(1)*  %e) gc "statepoint-example" {
; CHECK-VREG-LABEL: name:            test_limit
; CHECK-VREG:    %4:gr64 = COPY $r8
; CHECK-VREG:    %3:gr64 = COPY $rcx
; CHECK-VREG:    %2:gr64 = COPY $rdx
; CHECK-VREG:    %1:gr64 = COPY $rsi
; CHECK-VREG:    %0:gr64 = COPY $rdi
; CHECK-VREG:    MOV64mr %stack.0, 1, $noreg, 0, $noreg, %0 :: (store 8 into %stack.0)
; CHECK-VREG:    %5:gr64, %6:gr64, %7:gr64, %8:gr64 = STATEPOINT 0, 0, 0, @func, 2, 0, 2, 0, 2, 0, 2, 5, %4(tied-def 0), %3(tied-def 1), %2(tied-def 2), %1(tied-def 3), 1, 8, %stack.0, 0, 2, 0, 2, 5, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, csr_64, implicit-def $rsp, implicit-def $ssp :: (volatile load store 8 on %stack.0)
; CHECK-VREG:    %9:gr64 = MOV64rm %stack.0, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)
; CHECK-VREG:    $rdi = COPY %9
; CHECK-VREG:    $rsi = COPY %8
; CHECK-VREG:    $rdx = COPY %7
; CHECK-VREG:    $rcx = COPY %6
; CHECK-VREG:    $r8 = COPY %5
; CHECK-VREG:    CALL64pcrel32 @consume5, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit $rdx, implicit $rcx, implicit $r8, implicit-def $rsp, implicit-def $ssp
; CHECK-VREG:    RET 0
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c, i32 addrspace(1)* %d, i32 addrspace(1)* %e)]
  %rel1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %rel2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 1, i32 1)
  %rel3 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 2, i32 2)
  %rel4 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 3, i32 3)
  %rel5 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 4, i32 4)
  call void @consume5(i32 addrspace(1)* %rel1, i32 addrspace(1)* %rel2, i32 addrspace(1)* %rel3, i32 addrspace(1)* %rel4, i32 addrspace(1)* %rel5)
  ret void
}

; Test that CopyFromReg emitted during ISEL processing of gc.relocate are properly ordered w.r.t. statepoint.
define i8 addrspace(1)* @test_isel_sched(i8 addrspace(1)* %0, i8 addrspace(1)* %1, i32 %2) gc "statepoint-example" {
;CHECK-VREG-LABEL: name:            test_isel_sched
;CHECK-VREG:  bb.0.entry:
;CHECK-VREG:        %2:gr32 = COPY $edx
;CHECK-VREG:        %1:gr64 = COPY $rsi
;CHECK-VREG:        %0:gr64 = COPY $rdi
;CHECK-VREG:        TEST32rr %2, %2, implicit-def $eflags
;CHECK-VREG:        %5:gr64 = CMOV64rr %1, %0, 4, implicit $eflags
;CHECK-VREG:        %6:gr32 = MOV32r0 implicit-def dead $eflags
;CHECK-VREG:        %7:gr64 = SUBREG_TO_REG 0, killed %6, %subreg.sub_32bit
;CHECK-VREG:        $rdi = COPY %7
;CHECK-VREG:        $rsi = COPY %5
;CHECK-VREG:        %3:gr64, %4:gr64 = STATEPOINT 10, 0, 2, @bar, $rdi, $rsi, 2, 0, 2, 0, 2, 0, 2, 2, %1(tied-def 0), %0(tied-def 1), 2, 0, 2, 2, 0, 0, 1, 1, csr_64, implicit-def $rsp, implicit-def $ssp
;CHECK-VREG:        TEST32rr %2, %2, implicit-def $eflags
;CHECK-VREG:        %8:gr64 = CMOV64rr %3, %4, 4, implicit $eflags
;CHECK-VREG:        $rax = COPY %8
;CHECK-VREG:        RET 0, $rax
entry:
  %cmp = icmp eq i32 %2, 0
  %ptr = select i1 %cmp, i8 addrspace(1)* %0, i8 addrspace(1)* %1
  %token = call token (i64, i32, void (i8 addrspace(1)*, i8 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i8p1i8f(i64 10, i32 0, void (i8 addrspace(1)*, i8 addrspace(1)*)* @bar, i32 2, i32 0, i8 addrspace(1)* null, i8 addrspace(1)* %ptr, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* %0, i8 addrspace(1)* %1) ]
  %rel0 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token, i32 0, i32 0)
  %rel1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token, i32 1, i32 1)
  %res = select i1 %cmp, i8 addrspace(1)* %rel0, i8 addrspace(1)* %rel1
  ret i8 addrspace(1)* %res
}

declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare dso_local i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
declare dso_local i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token, i32, i32)
declare dso_local i1 @llvm.experimental.gc.result.i1(token)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidp1i8p1i8f(i64 immarg, i32 immarg, void (i8 addrspace(1)*, i8 addrspace(1)*)*, i32 immarg, i32 immarg, ...)


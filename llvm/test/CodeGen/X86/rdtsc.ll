; RUN: llc < %s -march=x86-64 -mcpu=generic | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=generic | FileCheck %s --check-prefix=CHECK --check-prefix=X86

; Verify that we correctly lower ISD::READCYCLECOUNTER.


define i64 @test_builtin_readcyclecounter() {
  %1 = tail call i64 @llvm.readcyclecounter()
  ret i64 %1
}
; CHECK-LABEL: test_builtin_readcyclecounter
; CHECK: rdtsc
; X86-NOT: shlq
; X86-NOT: or
; CHECK-NOT: mov
; CHECK: ret


; Verify that we correctly lower the Read Cycle Counter GCC x86 builtins
; (i.e. RDTSC and RDTSCP).

define i64 @test_builtin_rdtsc() {
  %1 = tail call i64 @llvm.x86.rdtsc()
  ret i64 %1
}
; CHECK-LABEL: test_builtin_rdtsc
; CHECK: rdtsc
; X86-NOT: shlq
; X86-NOT: or
; CHECK-NOT: mov
; CHECK: ret


define i64 @test_builtin_rdtscp(i8* %A) {
  %1 = tail call i64 @llvm.x86.rdtscp(i8* %A)
  ret i64 %1
}
; CHECK-LABEL: test_builtin_rdtscp
; CHECK: rdtscp
; X86-NOT: shlq
; CHECK:   movl	%ecx, (%{{[a-z0-9]+}})
; X86-NOT: shlq
; CHECK: ret


declare i64 @llvm.readcyclecounter()
declare i64 @llvm.x86.rdtscp(i8*)
declare i64 @llvm.x86.rdtsc()


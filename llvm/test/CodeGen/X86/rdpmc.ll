; RUN: llc < %s -march=x86-64 -mcpu=generic | FileCheck %s --check-prefix=CHECK --check-prefix=X86-64
; RUN: llc < %s -march=x86 -mcpu=generic | FileCheck %s --check-prefix=CHECK --check-prefix=X86

; Verify that we correctly lower the "Read Performance-Monitoring Counters"
; x86 builtin.


define i64 @test_builtin_read_pmc(i32 %ID) {
  %1 = tail call i64 @llvm.x86.rdpmc(i32 %ID)
  ret i64 %1
}
; CHECK-LABEL: test_builtin_read_pmc
; CHECK: rdpmc
; X86-NOT: shlq
; X86-NOT: or
; X86-64: shlq
; X86-64: or
; CHECK-NOT: mov
; CHECK: ret

declare i64 @llvm.x86.rdpmc(i32 %ID)


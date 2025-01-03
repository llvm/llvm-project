; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s

; RUN: llc --stop-after=prologepilog -o - %s \
; RUN: | llc -x=mir -enable-ipra -passes="module(require<reg-usage>,function(machine-function(reg-usage-collector)),print<reg-usage>)" -o /dev/null 2>&1 \
; RUN: | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Verify that bar does not clobber anything
; CHECK-NOT: bar Clobbered Registers:{{.+}}
; CHECK: bar Clobbered Registers:
define void @bar() #0 {
  ret void
}

; Verifies that inline assembly is correctly handled by giving a list of clobbered registers
; CHECK: foo Clobbered Registers: $ah $al $ax $ch $cl $cx $di $dih $dil $eax $ecx $edi $hax $hcx $hdi $rax $rcx $rdi
define void @foo() #0 {
  call void asm sideeffect "", "~{eax},~{ecx},~{edi}"() #0
  ret void
}

@llvm.used = appending global [2 x ptr] [ptr @foo, ptr @bar]

attributes #0 = { nounwind }

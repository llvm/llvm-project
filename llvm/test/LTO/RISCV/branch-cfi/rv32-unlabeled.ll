; RUN: llvm-as %s -o %t.main.bc
; RUN: llvm-as %p/Inputs/rv32-foo-unlabeled.ll -o %t.foo.unlabeled.bc
; RUN: llvm-link %t.main.bc %t.foo.unlabeled.bc -S | FileCheck --check-prefix=UNLABELED %s

; RUN: llvm-as %p/Inputs/rv32-foo-disabled.ll -o %t.foo.disabled.bc
; RUN: llvm-link %t.main.bc %t.foo.disabled.bc -S | FileCheck --check-prefix=DISABLED %s

; RUN: llvm-as %p/Inputs/rv32-foo-unknown-scheme.ll -o %t.foo.unknown.scheme.bc
; RUN: not llvm-link %t.main.bc %t.foo.unknown.scheme.bc 2>&1 | FileCheck --check-prefix=SCHEME-CONFLICT %s

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 @foo()
  ret i32 %call
}

declare dso_local i32 @foo() #1

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit" }

!llvm.module.flags = !{!0, !1, !2, !4, !5, !6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{i32 6, !"riscv-isa", !3}
!3 = !{!"rv32i2p1"}
; UNLABELED-DAG: [[P_FLAG:![0-9]+]] = !{i32 8, !"cf-protection-branch", i32 1}
; DISABLED-DAG: [[P_FLAG:![0-9]+]] = !{i32 8, !"cf-protection-branch", i32 0}
!4 = !{i32 8, !"cf-protection-branch", i32 1}
; UNLABELED-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
!5 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
!6 = !{i32 8, !"SmallDataLimit", i32 0}
; UNLABELED-DAG: !llvm.module.flags = !{{[{].*}}[[P_FLAG]]{{, .*}}[[S_FLAG]]{{[,}]}}
; DISABLED-DAG: !llvm.module.flags = !{{[{].*}}[[P_FLAG]]{{[,}]}}

; SCHEME-CONFLICT: error: linking module flags 'cf-branch-label-scheme': IDs have conflicting values: '!"unknown-scheme"' from {{.*}}, and '!"unlabeled"' from llvm-link

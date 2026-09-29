; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s --check-prefix=WINDOWS
; RUN: llc < %s -mtriple=aarch64-linux | FileCheck %s --check-prefix=LINUX

define dso_local void @b() #0 {
entry:
  br label %for.cond

for.cond:
  tail call void @a()
  br label %for.cond
}

declare dso_local void @a(...)

attributes #0 = { noreturn nounwind uwtable "tune-cpu"="cortex-a53" }

; LINUX-LABEL: b:
; LINUX: .p2align 4

; WINDOWS-LABEL: b:
; WINDOWS-NOT: .p2align

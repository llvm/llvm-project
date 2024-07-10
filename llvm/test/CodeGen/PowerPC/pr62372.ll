; RUN: llc -ppc-asm-full-reg-names -mcpu=pwr10 -mtriple powerpc64le-unknown-linux-gnu \
; RUN: -o - %s | FileCheck %s

@bar = dso_local global i32 0, align 4

define dso_local ptr @foo() #0 {
entry:
  ret ptr @bar
}

attributes #0 = { "use-soft-float"="true" }

; CHECK: paddi r3, 0, bar@PCREL, 1

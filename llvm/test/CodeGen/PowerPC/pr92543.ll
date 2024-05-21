; RUN: llc -ppc-asm-full-reg-names -o - %s| FileCheck %s

target triple = "powerpc64le-unknown-linux-gnu"

@bar = dso_local global i32 0, align 4

define dso_local ptr @foo() #0 {
entry:
  ret ptr @bar
}

attributes #0 = { noinline nounwind optnone uwtable "target-cpu"="pwr10" "target-features"="+altivec,+isa-v30-instructions,+isa-v31-instructions,+pcrelative-memops,+power10-vector,+power9-vector,+prefix-instrs,+vsx,-hard-float" "use-soft-float"="true" }


; CHECK: paddi r3, 0, bar@PCREL, 1

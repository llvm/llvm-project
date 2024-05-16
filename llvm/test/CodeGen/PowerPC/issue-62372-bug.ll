; RUN: llc -o - %s| FileCheck %s

target triple = "powerpc64le-unknown-linux-gnu"

@bar = dso_local global i32 0, align 4

define dso_local ptr @foo() #0 {
entry:
	  ret ptr @bar
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr10" "target-features"="+altivec,+bpermd,+crbits,+crypto,+direct-move,+extdiv,+isa-v206-instructions,+isa-v207-instructions,+isa-v30-instructions,+isa-v31-instructions,+mma,+paired-vector-memops,+pcrelative-memops,+power10-vector,+power8-vector,+power9-vector,+prefix-instrs,+quadword-atomics,+vsx,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-hard-float,-htm,-privileged,-rop-protect,-spe" "use-soft-float"="true" }


; CHECK: paddi 3, 0, bar@PCREL, 1

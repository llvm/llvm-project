target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

@InterposableAliasWrite1 = linkonce dso_local alias void(ptr), ptr @Write1

@PreemptableAliasWrite1 = dso_preemptable alias void(ptr), ptr @Write1
@AliasToPreemptableAliasWrite1 = dso_local alias void(ptr), ptr @PreemptableAliasWrite1

@AliasWrite1 = dso_local alias void(ptr), ptr @Write1

@BitcastAliasWrite1 = dso_local alias void(ptr), ptr @Write1
@AliasToBitcastAliasWrite1 = dso_local alias void(ptr), ptr @BitcastAliasWrite1


define dso_local void @Write1(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret void
}

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

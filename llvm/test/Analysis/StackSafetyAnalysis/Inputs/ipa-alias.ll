target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@InterposableAliasWrite1 = linkonce dso_local alias void(i8*), void(i8*)* @Write1

@PreemptableAliasWrite1 = dso_preemptable alias void(i8*), void(i8*)* @Write1
@AliasToPreemptableAliasWrite1 = dso_local alias void(i8*), void(i8*)* @PreemptableAliasWrite1

@AliasWrite1 = dso_local alias void(i8*), void(i8*)* @Write1

@BitcastAliasWrite1 = dso_local alias void(i32*), bitcast (void(i8*)* @Write1 to void(i32*)*)
@AliasToBitcastAliasWrite1 = dso_local alias void(i8*), bitcast (void(i32*)* @BitcastAliasWrite1 to void(i8*)*)

define dso_local void @Write1(i8* %p) {
entry:
  store i8 0, i8* %p, align 1
  ret void
}

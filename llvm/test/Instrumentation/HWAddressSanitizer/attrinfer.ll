; Standard library functions get inferred attributes, some of which are not
; correct when building for HWASan.

; RUN: opt < %s -passes=hwasan -S | FileCheck %s --check-prefixes=CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare float @frexpf(float noundef, ptr nocapture noundef) local_unnamed_addr #0

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: write) "frame-pointer"="non-leaf" "hwasan-abi"="interceptor" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fix-cortex-a53-835769,+fp-armv8,+neon,+outline-atomics,+tagged-globals,+v8a" }

; CHECK-NOT: memory(argmem: write)
; CHECK: nobuiltin

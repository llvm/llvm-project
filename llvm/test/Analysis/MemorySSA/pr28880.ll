; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

; This testcase is reduced from SingleSource/Benchmarks/Misc/fbench.c
; It is testing to make sure that the MemorySSA use optimizer
; comes up with right answers when dealing with multiple MemoryLocations
; over different blocks. See PR28880 for more details.
@global = external hidden unnamed_addr global double, align 8
@global.1 = external hidden unnamed_addr global double, align 8

; Function Attrs: nounwind ssp uwtable
define hidden fastcc void @hoge() unnamed_addr #0 {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb
; These accesses should not conflict.
; CHECK:  1 = MemoryDef(liveOnEntry)
; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   store double undef, double* @global, align 8
  store double undef, double* @global, align 8
; CHECK:  MemoryUse(liveOnEntry)
; MemoryUse(liveOnEntry)
; CHECK-NEXT:   %tmp = load double, double* @global.1, align 8
  %tmp = load double, double* @global.1, align 8
  unreachable

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2
  br i1 undef, label %bb4, label %bb6

bb4:                                              ; preds = %bb3
; These accesses should conflict.
; CHECK:  2 = MemoryDef(liveOnEntry)
; 2 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   store double 0.000000e+00, double* @global.1, align 8
  store double 0.000000e+00, double* @global.1, align 8
; CHECK:  MemoryUse(2)
; MemoryUse(2)
; CHECK-NEXT:   %tmp5 = load double, double* @global.1, align 8
  %tmp5 = load double, double* @global.1, align 8
  unreachable

bb6:                                              ; preds = %bb3
  unreachable
}

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }


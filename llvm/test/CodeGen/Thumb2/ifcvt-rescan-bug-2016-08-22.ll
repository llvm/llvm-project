; RUN: llc -O2 -o - %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-unknown-linux-gnueabihf"

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare void @_ZNSaIcEC2Ev() unnamed_addr #0 align 2

declare void @_ZNSsC1EPKcRKSaIcE() unnamed_addr #0

; It isn't valid to If-Convert the following function, even though the calls
; are in common. The calls clobber the predicate info.
; CHECK: cbnz r{{[0-9]+}}, .LBB0_2
; CHECK: BB#1
; CHECK: .LBB0_2
; Function Attrs: nounwind
define hidden void @_ZN4llvm14DOTGraphTraitsIPNS_13ScheduleDAGMIEE17getEdgeAttributesEPKNS_5SUnitENS_13SUnitIteratorEPKNS_11ScheduleDAGE() #0 align 2 {
  br i1 undef, label %1, label %2

; <label>:1:                                      ; preds = %0
  call void @_ZNSaIcEC2Ev() #0
  call void @_ZNSsC1EPKcRKSaIcE()
  br label %3

; <label>:2:                                      ; preds = %0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* undef) #0
  call void @_ZNSaIcEC2Ev() #0
  br label %3

; <label>:3:                                      ; preds = %2, %1
  ret void
}

attributes #0 = { nounwind }

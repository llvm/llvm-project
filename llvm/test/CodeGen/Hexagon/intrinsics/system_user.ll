; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: dc00:
; CHECK: dcfetch
define void @dc00(i8* nocapture readonly %p) local_unnamed_addr #0 {
  tail call void @llvm.hexagon.prefetch(i8* %p)
  ret void
}

; CHECK-LABEL: dc01:
; CHECK: dccleana
define void @dc01(i8* nocapture readonly %p) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y2.dccleana(i8* %p)
  ret void
}

; CHECK-LABEL: dc02:
; CHECK: dccleaninva
define void @dc02(i8* nocapture readonly %p) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y2.dccleaninva(i8* %p)
  ret void
}

; CHECK-LABEL: dc03:
; CHECK: dcinva
define void @dc03(i8* nocapture readonly %p) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y2.dcinva(i8* %p)
  ret void
}

; CHECK-LABEL: dc04:
; CHECK: dczeroa
define void @dc04(i8* nocapture %p) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y2.dczeroa(i8* %p)
  ret void
}

; CHECK-LABEL: dc05:
; CHECK: l2fetch(r{{[0-9]+}},r{{[0-9]+}})
define void @dc05(i8* nocapture readonly %p, i32 %q) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y4.l2fetch(i8* %p, i32 %q)
  ret void
}

; CHECK-LABEL: dc06:
; CHECK: l2fetch(r{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define void @dc06(i8* nocapture readonly %p, i64 %q) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.Y5.l2fetch(i8* %p, i64 %q)
  ret void
}

declare void @llvm.hexagon.prefetch(i8* nocapture) #1
declare void @llvm.hexagon.Y2.dccleana(i8* nocapture readonly) #2
declare void @llvm.hexagon.Y2.dccleaninva(i8* nocapture readonly) #2
declare void @llvm.hexagon.Y2.dcinva(i8* nocapture readonly) #2
declare void @llvm.hexagon.Y2.dczeroa(i8* nocapture) #3
declare void @llvm.hexagon.Y4.l2fetch(i8* nocapture readonly, i32) #2
declare void @llvm.hexagon.Y5.l2fetch(i8* nocapture readonly, i64) #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
attributes #1 = { inaccessiblemem_or_argmemonly nounwind }
attributes #2 = { nounwind }
attributes #3 = { argmemonly nounwind writeonly }

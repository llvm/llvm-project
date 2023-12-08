; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: xh_sh
; CHECK: sath
; CHECK-NOT: sxth
define i32 @xh_sh(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.sath(i32 %x)
  %1 = tail call i32 @llvm.hexagon.A2.sxth(i32 %0)
  ret i32 %1
}

; CHECK-LABEL: xb_sb
; CHECK: satb
; CHECK-NOT: sxtb
define i32 @xb_sb(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.satb(i32 %x)
  %1 = tail call i32 @llvm.hexagon.A2.sxtb(i32 %0)
  ret i32 %1
}

; CHECK-LABEL: xuh_suh
; CHECK: satuh
; CHECK-NOT: zxth
define i32 @xuh_suh(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.satuh(i32 %x)
  %1 = tail call i32 @llvm.hexagon.A2.zxth(i32 %0)
  ret i32 %1
}

; CHECK-LABEL: xub_sub
; CHECK: satub
; CHECK-NOT: zxtb
define i32 @xub_sub(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.satub(i32 %x)
  %1 = tail call i32 @llvm.hexagon.A2.zxtb(i32 %0)
  ret i32 %1
}


declare i32 @llvm.hexagon.A2.sxtb(i32) #1
declare i32 @llvm.hexagon.A2.sxth(i32) #1
declare i32 @llvm.hexagon.A2.zxtb(i32) #1
declare i32 @llvm.hexagon.A2.zxth(i32) #1

declare i32 @llvm.hexagon.A2.satb(i32) #1
declare i32 @llvm.hexagon.A2.sath(i32) #1
declare i32 @llvm.hexagon.A2.satub(i32) #1
declare i32 @llvm.hexagon.A2.satuh(i32) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone }

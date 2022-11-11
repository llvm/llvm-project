; RUN: opt -passes=sroa -S < %s | FileCheck %s

target datalayout = "e-p:64:32-i64:32-v32:32-n32-S64"

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

; CHECK: @wombat
; CHECK-NOT: alloca
; CHECK: ret void
define void @wombat(<4 x float> %arg1) {
bb:
  %tmp = alloca <4 x float>, align 16
  call void @llvm.lifetime.start.p0(i64 16, ptr %tmp)
  store <4 x float> %arg1, ptr %tmp, align 16
  %tmp18 = load <3 x float>, ptr %tmp
  call void @llvm.lifetime.end.p0(i64 16, ptr %tmp)
  call void @wombat3(<3 x float> %tmp18)
  ret void
}

; Function Attrs: nounwind
declare void @wombat3(<3 x float>) #0

attributes #0 = { nounwind }

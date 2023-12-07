; RUN: llc < %s -mtriple=arm64-apple-ios -asm-verbose=false | FileCheck %s

define float @load0(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load0:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %tmp = load i16, ptr %a, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16.f32(i16 %tmp)
  ret float %tmp1
}

define double @load1(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load1:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %tmp = load i16, ptr %a, align 2
  %conv = tail call double @llvm.convert.from.fp16.f64(i16 %tmp)
  ret double %conv
}

define float @load2(ptr nocapture readonly %a, i32 %i) nounwind {
; CHECK-LABEL: load2:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, w1, sxtw #1]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %idxprom
  %tmp = load i16, ptr %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16.f32(i16 %tmp)
  ret float %tmp1
}

define double @load3(ptr nocapture readonly %a, i32 %i) nounwind {
; CHECK-LABEL: load3:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, w1, sxtw #1]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %idxprom
  %tmp = load i16, ptr %arrayidx, align 2
  %conv = tail call double @llvm.convert.from.fp16.f64(i16 %tmp)
  ret double %conv
}

define float @load4(ptr nocapture readonly %a, i64 %i) nounwind {
; CHECK-LABEL: load4:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %i
  %tmp = load i16, ptr %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16.f32(i16 %tmp)
  ret float %tmp1
}

define double @load5(ptr nocapture readonly %a, i64 %i) nounwind {
; CHECK-LABEL: load5:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %i
  %tmp = load i16, ptr %arrayidx, align 2
  %conv = tail call double @llvm.convert.from.fp16.f64(i16 %tmp)
  ret double %conv
}

define float @load6(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load6:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, #20]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 10
  %tmp = load i16, ptr %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16.f32(i16 %tmp)
  ret float %tmp1
}

define double @load7(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load7:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, #20]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 10
  %tmp = load i16, ptr %arrayidx, align 2
  %conv = tail call double @llvm.convert.from.fp16.f64(i16 %tmp)
  ret double %conv
}

define float @load8(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load8:
; CHECK-NEXT: ldur [[HREG:h[0-9]+]], [x0, #-20]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 -10
  %tmp = load i16, ptr %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16.f32(i16 %tmp)
  ret float %tmp1
}

define double @load9(ptr nocapture readonly %a) nounwind {
; CHECK-LABEL: load9:
; CHECK-NEXT: ldur [[HREG:h[0-9]+]], [x0, #-20]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16, ptr %a, i64 -10
  %tmp = load i16, ptr %arrayidx, align 2
  %conv = tail call double @llvm.convert.from.fp16.f64(i16 %tmp)
  ret double %conv
}

define void @store0(ptr nocapture %a, float %val) nounwind {
; CHECK-LABEL: store0:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str  h0, [x0]
; CHECK-NEXT: ret

  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %val)
  store i16 %tmp, ptr %a, align 2
  ret void
}

define void @store1(ptr nocapture %a, double %val) nounwind {
; CHECK-LABEL: store1:
; CHECK-NEXT: fcvt s0, d0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str  h0, [x0]
; CHECK-NEXT: ret

  %conv = fptrunc double %val to float
  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  store i16 %tmp, ptr %a, align 2
  ret void
}

define void @store2(ptr nocapture %a, i32 %i, float %val) nounwind {
; CHECK-LABEL: store2:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, w1, sxtw #1]
; CHECK-NEXT: ret

  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %val)
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %idxprom
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store3(ptr nocapture %a, i32 %i, double %val) nounwind {
; CHECK-LABEL: store3:
; CHECK-NEXT: fcvt s0, d0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, w1, sxtw #1]
; CHECK-NEXT: ret

  %conv = fptrunc double %val to float
  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %idxprom
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store4(ptr nocapture %a, i64 %i, float %val) nounwind {
; CHECK-LABEL: store4:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, x1, lsl #1]
; CHECK-NEXT: ret

  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %val)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %i
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store5(ptr nocapture %a, i64 %i, double %val) nounwind {
; CHECK-LABEL: store5:
; CHECK-NEXT: fcvt s0, d0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, x1, lsl #1]
; CHECK-NEXT: ret

  %conv = fptrunc double %val to float
  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 %i
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store6(ptr nocapture %a, float %val) nounwind {
; CHECK-LABEL: store6:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, #20]
; CHECK-NEXT: ret

  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %val)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 10
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store7(ptr nocapture %a, double %val) nounwind {
; CHECK-LABEL: store7:
; CHECK-NEXT: fcvt s0, d0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: str h0, [x0, #20]
; CHECK-NEXT: ret

  %conv = fptrunc double %val to float
  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 10
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store8(ptr nocapture %a, float %val) nounwind {
; CHECK-LABEL: store8:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: stur h0, [x0, #-20]
; CHECK-NEXT: ret

  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %val)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 -10
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

define void @store9(ptr nocapture %a, double %val) nounwind {
; CHECK-LABEL: store9:
; CHECK-NEXT: fcvt s0, d0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: stur h0, [x0, #-20]
; CHECK-NEXT: ret

  %conv = fptrunc double %val to float
  %tmp = tail call i16 @llvm.convert.to.fp16.f32(float %conv)
  %arrayidx = getelementptr inbounds i16, ptr %a, i64 -10
  store i16 %tmp, ptr %arrayidx, align 2
  ret void
}

declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone
declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare i16 @llvm.convert.to.fp16.f64(double) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone

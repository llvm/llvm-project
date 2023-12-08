; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: r[[R00:[0-9]+]]:[[R01:[0-9]+]] = combine(r0,r1)
; CHECK: r[[R02:[0-9]+]]:[[R03:[0-9]+]] = asl(r[[R00]]:[[R01]],#17)
define i32 @f0(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshl.i32(i32 %a0, i32 %a1, i32 17)
  ret i32 %v0
}

; CHECK-LABEL: f1:
; CHECK: r[[R10:[0-9]+]]:[[R11:[0-9]+]] = combine(r0,r1)
; CHECK: r[[R12:[0-9]+]]:[[R13:[0-9]+]] = asl(r[[R10]]:[[R11]],r2)
define i32 @f1(i32 %a0, i32 %a1, i32 %a2) #1 {
b0:
  %v0 = tail call i32 @llvm.fshl.i32(i32 %a0, i32 %a1, i32 %a2)
  ret i32 %v0
}

; CHECK-LABEL: f2:
; CHECK: r[[R20:[0-9]+]]:[[R21:[0-9]+]] = asl(r1:0,#17)
; CHECK: r[[R20]]:[[R21]] |= lsr(r3:2,#47)
define i64 @f2(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 17)
  ret i64 %v0
}

; CHECK-LABEL: f3:
; CHECK: r[[R30:[0-9]+]]:[[R31:[0-9]+]] = asl(r1:0,r4)
; CHECK: r[[R32:[0-9]+]] = sub(#64,r4)
; CHECK: r[[R30]]:[[R31]] |= lsr(r3:2,r[[R32]])
define i64 @f3(i64 %a0, i64 %a1, i64 %a2) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 %a2)
  ret i64 %v0
}

; CHECK-LABEL: f4:
; CHECK: r[[R40:[0-9]+]]:[[R41:[0-9]+]] = combine(r0,r1)
; CHECK: r[[R42:[0-9]+]]:[[R43:[0-9]+]] = lsr(r[[R40]]:[[R41]],#17)
define i32 @f4(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshr.i32(i32 %a0, i32 %a1, i32 17)
  ret i32 %v0
}

; CHECK-LABEL: f5:
; CHECK: r[[R50:[0-9]+]]:[[R51:[0-9]+]] = combine(r0,r1)
; CHECK: r[[R52:[0-9]+]]:[[R53:[0-9]+]] = lsr(r[[R50]]:[[R51]],r2)
define i32 @f5(i32 %a0, i32 %a1, i32 %a2) #1 {
b0:
  %v0 = tail call i32 @llvm.fshr.i32(i32 %a0, i32 %a1, i32 %a2)
  ret i32 %v0
}

; CHECK-LABEL: f6:
; CHECK: r[[R60:[0-9]+]]:[[R61:[0-9]+]] = lsr(r3:2,#17)
; CHECK: r[[R60]]:[[R61]] |= asl(r1:0,#47)
define i64 @f6(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 17)
  ret i64 %v0
}

; CHECK-LABEL: f7:
; CHECK: r[[R70:[0-9]+]]:[[R71:[0-9]+]] = lsr(r3:2,r4)
; CHECK: r[[R72:[0-9]+]] = sub(#64,r4)
; CHECK: r[[R70]]:[[R71]] |= asl(r1:0,r6)
define i64 @f7(i64 %a0, i64 %a1, i64 %a2) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 %a2)
  ret i64 %v0
}

; CHECK-LABEL: f8:
; CHECK: r[[R80:[0-9]+]] = rol(r0,#17)
define i32 @f8(i32 %a0) #1 {
b0:
  %v0 = tail call i32 @llvm.fshl.i32(i32 %a0, i32 %a0, i32 17)
  ret i32 %v0
}

; CHECK-LABEL: f9:
; CHECK: r[[R90:[0-9]+]]:[[R91:[0-9]+]] = combine(r0,r0)
; CHECK: r[[R92:[0-9]+]]:[[R93:[0-9]+]] = asl(r[[R90]]:[[R91]],r1)
define i32 @f9(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshl.i32(i32 %a0, i32 %a0, i32 %a1)
  ret i32 %v0
}

; CHECK-LABEL: f10:
; CHECK: r[[RA0:[0-9]+]]:[[RA1:[0-9]+]] = rol(r1:0,#17)
define i64 @f10(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a0, i64 17)
  ret i64 %v0
}

; CHECK-LABEL: f11:
; CHECK: r[[RB0:[0-9]+]]:[[RB1:[0-9]+]] = asl(r1:0,r2)
; CHECK: r[[RB2:[0-9]+]] = sub(#64,r2)
; CHECK: r[[RB0]]:[[RB1]] |= lsr(r1:0,r[[RB2]])
define i64 @f11(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a0, i64 %a1)
  ret i64 %v0
}

; CHECK-LABEL: f12:
; CHECK: r[[RC0:[0-9]+]] = rol(r0,#15)
define i32 @f12(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshr.i32(i32 %a0, i32 %a0, i32 17)
  ret i32 %v0
}

; CHECK-LABEL: f13:
; CHECK: r[[RD0:[0-9]+]]:[[RD1:[0-9]+]] = combine(r0,r0)
; CHECK: r[[RD2:[0-9]+]]:[[RD3:[0-9]+]] = lsr(r[[RD0]]:[[RD1]],r1)
define i32 @f13(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshr.i32(i32 %a0, i32 %a0, i32 %a1)
  ret i32 %v0
}

; CHECK-LABEL: f14:
; CHECK: r[[RE0:[0-9]+]]:[[RE1:[0-9]+]] = rol(r1:0,#47)
define i64 @f14(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a0, i64 17)
  ret i64 %v0
}

; CHECK-LABEL: f15:
; CHECK: r[[RF0:[0-9]+]]:[[RF1:[0-9]+]] = lsr(r1:0,r2)
; CHECK: r[[RF2:[0-9]+]] = sub(#64,r2)
; CHECK: r[[RF0]]:[[RF1]] |= asl(r1:0,r[[RF2]])
define i64 @f15(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a0, i64 %a1)
  ret i64 %v0
}

; CHECK-LABEL: f16:
; CHECK: r[[RG0:[0-9]+]]:[[RG1:[0-9]+]] = valignb(r1:0,r3:2,#7)
define i64 @f16(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 8)
  ret i64 %v0
}

; CHECK-LABEL: f17:
; CHECK: r[[RH0:[0-9]+]]:[[RH1:[0-9]+]] = valignb(r1:0,r3:2,#6)
define i64 @f17(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 16)
  ret i64 %v0
}

; CHECK-LABEL: f18:
; CHECK: r[[RI0:[0-9]+]]:[[RI1:[0-9]+]] = valignb(r1:0,r3:2,#5)
define i64 @f18(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 24)
  ret i64 %v0
}

; CHECK-LABEL: f19:
; CHECK: r[[RJ0:[0-9]+]]:[[RJ1:[0-9]+]] = valignb(r1:0,r3:2,#4)
define i64 @f19(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 32)
  ret i64 %v0
}

; CHECK-LABEL: f20:
; CHECK: r[[RK0:[0-9]+]]:[[RK1:[0-9]+]] = valignb(r1:0,r3:2,#3)
define i64 @f20(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 40)
  ret i64 %v0
}

; CHECK-LABEL: f21:
; CHECK: r[[RL0:[0-9]+]]:[[RL1:[0-9]+]] = valignb(r1:0,r3:2,#2)
define i64 @f21(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 48)
  ret i64 %v0
}

; CHECK-LABEL: f22:
; CHECK: r[[RM0:[0-9]+]]:[[RM1:[0-9]+]] = valignb(r1:0,r3:2,#1)
define i64 @f22(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshl.i64(i64 %a0, i64 %a1, i64 56)
  ret i64 %v0
}

; CHECK-LABEL: f23:
; CHECK: r[[RN0:[0-9]+]]:[[RN1:[0-9]+]] = valignb(r1:0,r3:2,#1)
define i64 @f23(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 8)
  ret i64 %v0
}

; CHECK-LABEL: f24:
; CHECK: r[[RO0:[0-9]+]]:[[RO1:[0-9]+]] = valignb(r1:0,r3:2,#2)
define i64 @f24(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 16)
  ret i64 %v0
}

; CHECK-LABEL: f25:
; CHECK: r[[RP0:[0-9]+]]:[[RP1:[0-9]+]] = valignb(r1:0,r3:2,#3)
define i64 @f25(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 24)
  ret i64 %v0
}

; CHECK-LABEL: f26:
; CHECK: r[[RQ0:[0-9]+]]:[[RQ1:[0-9]+]] = valignb(r1:0,r3:2,#4)
define i64 @f26(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 32)
  ret i64 %v0
}

; CHECK-LABEL: f27:
; CHECK: r[[RR0:[0-9]+]]:[[RR1:[0-9]+]] = valignb(r1:0,r3:2,#5)
define i64 @f27(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 40)
  ret i64 %v0
}

; CHECK-LABEL: f28:
; CHECK: r[[RS0:[0-9]+]]:[[RS1:[0-9]+]] = valignb(r1:0,r3:2,#6)
define i64 @f28(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 48)
  ret i64 %v0
}

; CHECK-LABEL: f29:
; CHECK: r[[RT0:[0-9]+]]:[[RT1:[0-9]+]] = valignb(r1:0,r3:2,#7)
define i64 @f29(i64 %a0, i64 %a1) #1 {
b0:
  %v0 = tail call i64 @llvm.fshr.i64(i64 %a0, i64 %a1, i64 56)
  ret i64 %v0
}

; CHECK-LABEL: f30:
; CHECK: r[[R00:[0-9]+]] = combine(r0.l,r1.h)
define i32 @f30(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshl.i32(i32 %a0, i32 %a1, i32 16)
  ret i32 %v0
}

; CHECK-LABEL: f31:
; CHECK: r[[R00:[0-9]+]] = combine(r0.l,r1.h)
define i32 @f31(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = tail call i32 @llvm.fshr.i32(i32 %a0, i32 %a1, i32 16)
  ret i32 %v0
}

declare i32 @llvm.fshl.i32(i32, i32, i32) #0
declare i32 @llvm.fshr.i32(i32, i32, i32) #0
declare i64 @llvm.fshl.i64(i64, i64, i64) #0
declare i64 @llvm.fshr.i64(i64, i64, i64) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="-packets" }

; RUN: llc < %s -O0 -mtriple=armv7-eabi -mcpu=cortex-a8 -mattr=+neon,+fp-armv8 | FileCheck %s
; RUN: llc < %s -O3 -mtriple=armv7-eabi -mcpu=cortex-a8 -mattr=+neon,+fp-armv8 | FileCheck %s

@a = common global double 0.000000e+00, align 8

; Function Attrs: noinline nounwind uwtable
define void @strtod() {
entry:
  ; CHECK: vmrs r{{[0-9]+}}, fpscr
  %0 = call i32 @llvm.get.rounding()
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store double 5.000000e-01, ptr @a, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind
define void @fn1(ptr nocapture %p) local_unnamed_addr {
entry:
  ; CHECK: vmrs r{{[0-9]+}}, fpscr
  %0 = tail call i32 @llvm.arm.get.fpscr()
  store i32 %0, ptr %p, align 4
  ; CHECK: vmsr fpscr, r{{[0-9]+}}
  tail call void @llvm.arm.set.fpscr(i32 1)
  ; CHECK: vmrs r{{[0-9]+}}, fpscr
  %1 = tail call i32 @llvm.arm.get.fpscr()
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i32 1
  store i32 %1, ptr %arrayidx1, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @llvm.arm.get.fpscr()

; Function Attrs: nounwind writeonly
declare void @llvm.arm.set.fpscr(i32)

; Function Attrs: nounwind
declare i32 @llvm.get.rounding()

; RUN: llc -O3 -mtriple=x86_64-pc-linux < %s -stop-after=finalize-isel | FileCheck %s

define i32 @f20u(double %x) #0 {
; CHECK-LABEL: name: f20u
; CHECK: liveins: $xmm0
; CHECK: [[COPY:%[0-9]+]]:fr64 = COPY $xmm0
; CHECK: [[CVTTSD2SI64rr:%[0-9]+]]:gr64 = CVTTSD2SI64rr [[COPY]], implicit $mxcsr
; CHECK: [[COPY1:%[0-9]+]]:gr32 = COPY [[CVTTSD2SI64rr]].sub_32bit
; CHECK: $eax = COPY [[COPY1]]
; CHECK: RET 0, $eax
entry:
  %result = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %result
}

attributes #0 = { strictfp }

declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)

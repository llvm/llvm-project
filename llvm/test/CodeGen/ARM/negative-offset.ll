; RUN: llc -mtriple=arm-eabi -O3 %s -o - | FileCheck %s

; Function Attrs: nounwind readonly
define arm_aapcscc i32 @sum(ptr nocapture readonly %p) #0 {
entry:
;CHECK-LABEL: sum:
;CHECK-NOT: sub
;CHECK: ldr r{{.*}}, [r0, #-16]
;CHECK: ldr r{{.*}}, [r0, #-8]
  %arrayidx = getelementptr inbounds i32, ptr %p, i32 -4
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i32 -2
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}


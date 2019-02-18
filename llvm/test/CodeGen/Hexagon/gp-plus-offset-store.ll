; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate store instructions with global + offset

%s.0 = type { i8, i8, i16, i32 }

@g0 = common global %s.0 zeroinitializer, align 4

; CHECK-LABEL: f0:
; CHECK: memb(##g0+1) = r{{[0-9]+}}
define void @f0(i32 %a0, i32 %a1, i8 zeroext %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  store i8 %a2, i8* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 1), align 1
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; CHECK-LABEL: f1:
; CHECK: memh(##g0+2) = r{{[0-9]+}}
define void @f1(i32 %a0, i32 %a1, i16 signext %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  store i16 %a2, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 2
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }

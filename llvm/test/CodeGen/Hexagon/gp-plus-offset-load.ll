; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate load instructions with global + offset


%s.0 = type { i8, i8, i16, i32 }

@g0 = common global %s.0 zeroinitializer, align 4

; CHECK-LABEL: f0:
; CHECK: r{{[0-9]+}} = memw(##g0+4)
define void @f0(i32 %a0, i32 %a1, ptr nocapture %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = load i32, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 3), align 4
  store i32 %v1, ptr %a2, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]+}} = memub(##g0+1)
define void @f1(i32 %a0, i32 %a1, ptr nocapture %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 1), align 1
  store i8 %v1, ptr %a2, align 1
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; CHECK-LABEL: f2:
; CHECK: r{{[0-9]+}} = memuh(##g0+2)
define void @f2(i32 %a0, i32 %a1, ptr %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = load i16, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 2), align 2
  store i16 %v1, ptr %a2, align 2
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }

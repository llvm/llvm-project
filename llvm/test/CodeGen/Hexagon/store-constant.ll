; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Generate stores with assignment of constant values.

; CHECK: memw{{.*}} = {{.*}}#0
; CHECK: memw{{.*}} = {{.*}}#1
; CHECK: memh{{.*}} = {{.*}}#2
; CHECK: memh{{.*}} = {{.*}}#3
; CHECK: memb{{.*}} = {{.*}}#4
; CHECK: memb{{.*}} = {{.*}}#5

define void @f0(ptr nocapture %a0) #0 {
b0:
  store i32 0, ptr %a0, align 4
  ret void
}

define void @f1(ptr nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i32, ptr %a0, i32 1
  store i32 1, ptr %v0, align 4
  ret void
}

define void @f2(ptr nocapture %a0) #0 {
b0:
  store i16 2, ptr %a0, align 2
  ret void
}

define void @f3(ptr nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i16, ptr %a0, i32 2
  store i16 3, ptr %v0, align 2
  ret void
}

define void @f4(ptr nocapture %a0) #0 {
b0:
  store i8 4, ptr %a0, align 1
  ret void
}

define void @f5(ptr nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i8, ptr %a0, i32 2
  store i8 5, ptr %v0, align 1
  ret void
}

attributes #0 = { nounwind }

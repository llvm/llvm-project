; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Generate stores with assignment of constant values.

; CHECK: memw{{.*}} = {{.*}}#0
; CHECK: memw{{.*}} = {{.*}}#1
; CHECK: memh{{.*}} = {{.*}}#2
; CHECK: memh{{.*}} = {{.*}}#3
; CHECK: memb{{.*}} = {{.*}}#4
; CHECK: memb{{.*}} = {{.*}}#5

define void @f0(i32* nocapture %a0) #0 {
b0:
  store i32 0, i32* %a0, align 4
  ret void
}

define void @f1(i32* nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i32, i32* %a0, i32 1
  store i32 1, i32* %v0, align 4
  ret void
}

define void @f2(i16* nocapture %a0) #0 {
b0:
  store i16 2, i16* %a0, align 2
  ret void
}

define void @f3(i16* nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i16, i16* %a0, i32 2
  store i16 3, i16* %v0, align 2
  ret void
}

define void @f4(i8* nocapture %a0) #0 {
b0:
  store i8 4, i8* %a0, align 1
  ret void
}

define void @f5(i8* nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i8, i8* %a0, i32 2
  store i8 5, i8* %v0, align 1
  ret void
}

attributes #0 = { nounwind }

; RUN: llc < %s -mtriple=thumbv7-apple-ios5.0.0 | FileCheck %s
; rdar://10464621

; DAG combine increases loads from packed types. ARM load / store optimizer then
; combined them into a ldm which causes runtime exception.

%struct.InformationBlock = type <{ i32, %struct.FlagBits, %struct.FlagBits }>
%struct.FlagBits = type <{ [4 x i32] }>

@infoBlock = external global %struct.InformationBlock

define hidden void @foo() {
; CHECK-LABEL: foo:
; CHECK: ldr.w
; CHECK: ldr.w
; CHECK-NOT: ldm
entry:
  %tmp13 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 1, i32 0, i32 0), align 1
  %tmp15 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 1, i32 0, i32 1), align 1
  %tmp17 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 1, i32 0, i32 2), align 1
  %tmp19 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 1, i32 0, i32 3), align 1
  %tmp = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 2, i32 0, i32 0), align 1
  %tmp3 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 2, i32 0, i32 1), align 1
  %tmp4 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 2, i32 0, i32 2), align 1
  %tmp5 = load i32, ptr getelementptr inbounds (%struct.InformationBlock, ptr @infoBlock, i32 0, i32 2, i32 0, i32 3), align 1
  %insert21 = insertvalue [4 x i32] undef, i32 %tmp13, 0
  %insert23 = insertvalue [4 x i32] %insert21, i32 %tmp15, 1
  %insert25 = insertvalue [4 x i32] %insert23, i32 %tmp17, 2
  %insert27 = insertvalue [4 x i32] %insert25, i32 %tmp19, 3
  %insert = insertvalue [4 x i32] undef, i32 %tmp, 0
  %insert7 = insertvalue [4 x i32] %insert, i32 %tmp3, 1
  %insert9 = insertvalue [4 x i32] %insert7, i32 %tmp4, 2
  %insert11 = insertvalue [4 x i32] %insert9, i32 %tmp5, 3
  tail call void @bar([4 x i32] %insert27, [4 x i32] %insert11)
  ret void
}

declare void @bar([4 x i32], [4 x i32])

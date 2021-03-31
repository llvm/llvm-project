; RUN: opt < %s -basic-aa -dse -S | FileCheck %s

; CHECK-LABEL: void @skipBarrier(i8* %ptr)
define void @skipBarrier(i8* %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, i8* %ptr
; CHECK: %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
  %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; CHECK: store i8 43
  store i8 43, i8* %ptr2
  ret void
}

; CHECK-LABEL: void @skip2Barriers(i8* %ptr)
define void @skip2Barriers(i8* %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, i8* %ptr
; CHECK: %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
  %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; CHECK-NOT: store i8 43
  store i8 43, i8* %ptr2
  %ptr3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr2)
  %ptr4 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr3)

; CHECK: store i8 44
  store i8 44, i8* %ptr4
  ret void
}

; CHECK-LABEL: void @skip3Barriers(i8* %ptr)
define void @skip3Barriers(i8* %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, i8* %ptr
; CHECK: %ptr2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr)
  %ptr2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr)
; CHECK-NOT: store i8 43
  store i8 43, i8* %ptr2
  %ptr3 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr2)
  %ptr4 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr3)

; CHECK: store i8 44
  store i8 44, i8* %ptr4
  ret void
}

; CHECK-LABEL: void @skip4Barriers(i8* %ptr)
define void @skip4Barriers(i8* %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, i8* %ptr
; CHECK: %ptr2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr)
  %ptr2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr)
; CHECK-NOT: store i8 43
  store i8 43, i8* %ptr2
  %ptr3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr2)
  %ptr4 = call i8* @llvm.strip.invariant.group.p0i8(i8* %ptr3)
  %ptr5 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr3)

; CHECK: store i8 44
  store i8 44, i8* %ptr5
  ret void
}


declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)

; RUN: opt < %s -passes=dse -S | FileCheck %s

; CHECK-LABEL: void @skipBarrier(ptr %ptr)
define void @skipBarrier(ptr %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, ptr %ptr
; CHECK: %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
  %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: store i8 43
  store i8 43, ptr %ptr2
  ret void
}

; CHECK-LABEL: void @skip2Barriers(ptr %ptr)
define void @skip2Barriers(ptr %ptr) {
; CHECK-NOT: store i8 42
  store i8 42, ptr %ptr
; CHECK: %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
  %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK-NOT: store i8 43
  store i8 43, ptr %ptr2
  %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr2)
  %ptr4 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr3)

; CHECK: store i8 44
  store i8 44, ptr %ptr4
  ret void
}


declare ptr @llvm.launder.invariant.group.p0(ptr)

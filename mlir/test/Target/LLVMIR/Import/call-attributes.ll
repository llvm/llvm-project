; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

%struct.S = type { i8 }

; CHECK-LABEL: @t
define void @t(i1 %0) #0 {
  %3 = alloca %struct.S, align 1
  ; CHECK-NOT: llvm.call @z(%1) {no_unwind} : (!llvm.ptr) -> ()
  ; CHECK: llvm.call @z(%1) : (!llvm.ptr) -> ()
  call void @z(ptr %3)
  ret void
}

define linkonce_odr void @z(ptr %0) #0 {
  ret void
}

attributes #0 = { nounwind }

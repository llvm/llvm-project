; RUN: llvm-as < %s | llvm-dis

%S = type { i32, i32 }

define void @simple_scalar_allocation() {
entry:
; CHECK: %ptr = call elementtype(i32) ptr @llvm.structured.alloca.p0()
  %ptr = call elementtype(i32) ptr @llvm.structured.alloca()
  ret void
}

define void @struct_allocation() {
entry:
; CHECK: %ptr = call elementtype(%S) ptr @llvm.structured.alloca.p0()
  %ptr = call elementtype(%S) ptr @llvm.structured.alloca()
  ret void
}

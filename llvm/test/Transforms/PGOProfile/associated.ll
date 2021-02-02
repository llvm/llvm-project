; RUN: opt < %s -pgo-instr-gen -instrprof -counter-link-order -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -counter-link-order -S | FileCheck %s

; CHECK: @__profc_foo = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8, !associated !0
; CHECK: @__profd_foo = private global {{.*}}, section "__llvm_prf_data", align 8, !associated !1

define void @foo() {
  ret void
}

; CHECK: !0 = !{void ()* @foo}
; CHECK: !1 = !{[1 x i64]* @__profc_foo}

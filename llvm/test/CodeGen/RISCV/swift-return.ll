; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,RV64,RV64-SOFT
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi=lp64d -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,RV64,RV64D

; The Swift calling convention returns values directly in up to four
; registers, twice the limit of the standard convention.

; A four-element aggregate is returned directly in a0-a3.
; CHECK-LABEL: gen4:
; CHECK-DAG: mv a1, a0
; CHECK-DAG: mv a2, a0
; CHECK-DAG: mv a3, a0
; CHECK: ret
define swiftcc { i32, i32, i32, i32 } @gen4(i32 %key) {
  %v0 = insertvalue { i32, i32, i32, i32 } undef, i32 %key, 0
  %v1 = insertvalue { i32, i32, i32, i32 } %v0, i32 %key, 1
  %v2 = insertvalue { i32, i32, i32, i32 } %v1, i32 %key, 2
  %v3 = insertvalue { i32, i32, i32, i32 } %v2, i32 %key, 3
  ret { i32, i32, i32, i32 } %v3
}

; CHECK-LABEL: call_gen4:
; CHECK: call gen4
; CHECK-DAG: add{{w?}} {{.*}}, a0, a1
; CHECK-DAG: add{{w?}} {{.*}}, a2, a3
; CHECK: ret
define i32 @call_gen4(i32 %key) {
  %res = call swiftcc { i32, i32, i32, i32 } @gen4(i32 %key)
  %v0 = extractvalue { i32, i32, i32, i32 } %res, 0
  %v1 = extractvalue { i32, i32, i32, i32 } %res, 1
  %v2 = extractvalue { i32, i32, i32, i32 } %res, 2
  %v3 = extractvalue { i32, i32, i32, i32 } %res, 3
  %s0 = add i32 %v0, %v1
  %s1 = add i32 %v2, %v3
  %s2 = add i32 %s0, %s1
  ret i32 %s2
}

; The same aggregate under the C calling convention is still returned
; indirectly: the four-register return is Swift-only.
; CHECK-LABEL: gen4_ccc:
; CHECK-DAG: sw {{.*}}, 0(a0)
; CHECK-DAG: sw {{.*}}, 4(a0)
; CHECK-DAG: sw {{.*}}, 8(a0)
; CHECK-DAG: sw {{.*}}, 12(a0)
; CHECK: ret
define { i32, i32, i32, i32 } @gen4_ccc(i32 %key) {
  %v0 = insertvalue { i32, i32, i32, i32 } undef, i32 %key, 0
  %v1 = insertvalue { i32, i32, i32, i32 } %v0, i32 %key, 1
  %v2 = insertvalue { i32, i32, i32, i32 } %v1, i32 %key, 2
  %v3 = insertvalue { i32, i32, i32, i32 } %v2, i32 %key, 3
  ret { i32, i32, i32, i32 } %v3
}

; A five-element aggregate exceeds the four-register budget and is demoted
; to an indirect (sret) return.
; CHECK-LABEL: gen5:
; CHECK-DAG: sw {{.*}}, 0(a0)
; CHECK-DAG: sw {{.*}}, 4(a0)
; CHECK-DAG: sw {{.*}}, 8(a0)
; CHECK-DAG: sw {{.*}}, 12(a0)
; CHECK-DAG: sw {{.*}}, 16(a0)
; CHECK: ret
define swiftcc { i32, i32, i32, i32, i32 } @gen5(i32 %key) {
  %v0 = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %v1 = insertvalue { i32, i32, i32, i32, i32 } %v0, i32 %key, 1
  %v2 = insertvalue { i32, i32, i32, i32, i32 } %v1, i32 %key, 2
  %v3 = insertvalue { i32, i32, i32, i32, i32 } %v2, i32 %key, 3
  %v4 = insertvalue { i32, i32, i32, i32, i32 } %v3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %v4
}

; A pair of i64s fits the budget on both rv32 (a0-a3 after splitting) and
; rv64 (a0-a1).
; CHECK-LABEL: gen2i64:
; RV32-DAG: mv a2, a0
; RV32-DAG: mv a3, a1
; RV64: mv a1, a0
; CHECK: ret
define swiftcc { i64, i64 } @gen2i64(i64 %key) {
  %v0 = insertvalue { i64, i64 } undef, i64 %key, 0
  %v1 = insertvalue { i64, i64 } %v0, i64 %key, 1
  ret { i64, i64 } %v1
}

; The mixed case from the swiftcall review: { i64, float, i64 } is returned
; in a0/fa0/a1 under a hard-float ABI and a0-a2 under a soft-float ABI
; (rv64).
; RV64-LABEL: gen_mixed:
; RV64-SOFT-DAG: mv a1, a0
; RV64-SOFT-DAG: mv a2, a0
; RV64D-DAG: fcvt.s.l fa0, a0
; RV64D-DAG: mv a1, a0
; RV64: ret
define swiftcc { i64, float, i64 } @gen_mixed(i64 %key) {
  %f = sitofp i64 %key to float
  %v0 = insertvalue { i64, float, i64 } undef, i64 %key, 0
  %v1 = insertvalue { i64, float, i64 } %v0, float %f, 1
  %v2 = insertvalue { i64, float, i64 } %v1, i64 %key, 2
  ret { i64, float, i64 } %v2
}

; Four doubles are returned in fa0-fa3 under lp64d.
; RV64D-LABEL: gen4f64:
; RV64D-DAG: fmv.d fa1, fa0
; RV64D-DAG: fmv.d fa2, fa0
; RV64D-DAG: fmv.d fa3, fa0
; RV64D: ret
define swiftcc { double, double, double, double } @gen4f64(double %key) {
  %v0 = insertvalue { double, double, double, double } undef, double %key, 0
  %v1 = insertvalue { double, double, double, double } %v0, double %key, 1
  %v2 = insertvalue { double, double, double, double } %v1, double %key, 2
  %v3 = insertvalue { double, double, double, double } %v2, double %key, 3
  ret { double, double, double, double } %v3
}

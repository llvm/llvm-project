; Tests for lrint and llrint, with both i32 and i64 checked.

; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=riscv32 | FileCheck %s --check-prefixes=RV32
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=riscv32 | FileCheck %s --check-prefixes=RV32
; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=riscv64 | FileCheck %s --check-prefixes=RV64
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=riscv64 | FileCheck %s --check-prefixes=RV64

; FIXME: crash
; define ITy @test_lrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.lrint.ITy.f16(half %x)
; }

; define ITy @test_llrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.llrint.ITy.f16(half %x)
; }

define ITy @test_lrint_ixx_f32(float %x) nounwind {
; RV32-LABEL: test_lrint_ixx_f32:
; RV32:         call lrintf
;
; RV64-LABEL: test_lrint_ixx_f32:
; RV64:         call lrintf
  %res = tail call ITy @llvm.lrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f32(float %x) nounwind {
; RV32-LABEL: test_llrint_ixx_f32:
; RV32:         call llrintf
;
; RV64-LABEL: test_llrint_ixx_f32:
; RV64:         call llrintf
  %res = tail call ITy @llvm.llrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_lrint_ixx_f64(double %x) nounwind {
; RV32-LABEL: test_lrint_ixx_f64:
; RV32:         call lrint
;
; RV64-LABEL: test_lrint_ixx_f64:
; RV64:         call lrint
  %res = tail call ITy @llvm.lrint.ITy.f64(double %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f64(double %x) nounwind {
; RV32-LABEL: test_llrint_ixx_f64:
; RV32:         call llrint
;
; RV64-LABEL: test_llrint_ixx_f64:
; RV64:         call llrint
  %res = tail call ITy @llvm.llrint.ITy.f64(double %x)
  ret ITy %res
}

; FIXME(#44744): incorrect libcall on riscv32
define ITy @test_lrint_ixx_f128(fp128 %x) nounwind {
; RV32-LABEL: test_lrint_ixx_f128:
; RV32:         call lrintl
;
; RV64-LABEL: test_lrint_ixx_f128:
; RV64:         call lrintl
  %res = tail call ITy @llvm.lrint.ITy.f128(fp128 %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f128(fp128 %x) nounwind {
; RV32-LABEL: test_llrint_ixx_f128:
; RV32:         call llrintl
;
; RV64-LABEL: test_llrint_ixx_f128:
; RV64:         call llrintl
  %res = tail call ITy @llvm.llrint.ITy.f128(fp128 %x)
  ret ITy %res
}

; Tests for lrint and llrint, with both i32 and i64 checked.

; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=loongarch32 | FileCheck %s --check-prefixes=LA32
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=loongarch32 | FileCheck %s --check-prefixes=LA32
; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=loongarch64 | FileCheck %s --check-prefixes=LA64-I32
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=loongarch64 | FileCheck %s --check-prefixes=LA64-I64

; FIXME: crash
; define ITy @test_lrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.lrint.ITy.f16(half %x)
;   ret ITy %res
; }

; define ITy @test_llrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.llrint.ITy.f16(half %x)
;   ret ITy %res
; }

define ITy @test_lrint_ixx_f32(float %x) nounwind {
; LA32-LABEL: test_lrint_ixx_f32:
; LA32:         bl lrintf
;
; LA64-I32-LABEL: test_lrint_ixx_f32:
; LA64-I32:         pcaddu18i $ra, %call36(lrintf)
;
; LA64-I64-LABEL: test_lrint_ixx_f32:
; LA64-I64:         pcaddu18i $t8, %call36(lrintf)
  %res = tail call ITy @llvm.lrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f32(float %x) nounwind {
; LA32-LABEL: test_llrint_ixx_f32:
; LA32:         bl llrintf
;
; LA64-I32-LABEL: test_llrint_ixx_f32:
; LA64-I32:         pcaddu18i $ra, %call36(llrintf)
;
; LA64-I64-LABEL: test_llrint_ixx_f32:
; LA64-I64:         pcaddu18i $t8, %call36(llrintf)
  %res = tail call ITy @llvm.llrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_lrint_ixx_f64(double %x) nounwind {
; LA32-LABEL: test_lrint_ixx_f64:
; LA32:         bl lrint
;
; LA64-I32-LABEL: test_lrint_ixx_f64:
; LA64-I32:         pcaddu18i $ra, %call36(lrint)
;
; LA64-I64-LABEL: test_lrint_ixx_f64:
; LA64-I64:         pcaddu18i $t8, %call36(lrint)
  %res = tail call ITy @llvm.lrint.ITy.f64(double %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f64(double %x) nounwind {
; LA32-LABEL: test_llrint_ixx_f64:
; LA32:         bl llrint
;
; LA64-I32-LABEL: test_llrint_ixx_f64:
; LA64-I32:         pcaddu18i $ra, %call36(llrint)
;
; LA64-I64-LABEL: test_llrint_ixx_f64:
; LA64-I64:         pcaddu18i $t8, %call36(llrint)
  %res = tail call ITy @llvm.llrint.ITy.f64(double %x)
  ret ITy %res
}

; FIXME(#44744): incorrect libcall on loongarch32
define ITy @test_lrint_ixx_f128(fp128 %x) nounwind {
; LA32-LABEL: test_lrint_ixx_f128:
; LA32:         bl lrintl
;
; LA64-I32-LABEL: test_lrint_ixx_f128:
; LA64-I32:         pcaddu18i $ra, %call36(lrintl)
;
; LA64-I64-LABEL: test_lrint_ixx_f128:
; LA64-I64:         pcaddu18i $ra, %call36(lrintl)
  %res = tail call ITy @llvm.lrint.ITy.f128(fp128 %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f128(fp128 %x) nounwind {
; LA32-LABEL: test_llrint_ixx_f128:
; LA32:         bl llrintl
;
; LA64-I32-LABEL: test_llrint_ixx_f128:
; LA64-I32:         pcaddu18i $ra, %call36(llrintl)
;
; LA64-I64-LABEL: test_llrint_ixx_f128:
; LA64-I64:         pcaddu18i $ra, %call36(llrintl)
  %res = tail call ITy @llvm.llrint.ITy.f128(fp128 %x)
  ret ITy %res
}

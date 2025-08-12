; Tests for lrint and llrint, with both i32 and i64 checked.

; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=sparc   | FileCheck %s --check-prefixes=SPARC32
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=sparc   | FileCheck %s --check-prefixes=SPARC32
; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=sparc64 | FileCheck %s --check-prefixes=SPARC64
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=sparc64 | FileCheck %s --check-prefixes=SPARC64

; FIXME: crash "Input type needs to be promoted!"
; define ITy @test_lrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.lrint.ITy.f16(half %x)
;   ret ITy %res
; }

; define ITy @test_llrint_ixx_f16(half %x) nounwind {
;   %res = tail call ITy @llvm.llrint.ITy.f16(half %x)
;   ret ITy %res
; }

define ITy @test_lrint_ixx_f32(float %x) nounwind {
; SPARC32-LABEL: test_lrint_ixx_f32:
; SPARC32:         call lrintf
;
; SPARC64-LABEL: test_lrint_ixx_f32:
; SPARC64:         call lrintf
  %res = tail call ITy @llvm.lrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f32(float %x) nounwind {
; SPARC32-LABEL: test_llrint_ixx_f32:
; SPARC32:         call llrintf
;
; SPARC64-LABEL: test_llrint_ixx_f32:
; SPARC64:         call llrintf
  %res = tail call ITy @llvm.llrint.ITy.f32(float %x)
  ret ITy %res
}

define ITy @test_lrint_ixx_f64(double %x) nounwind {
; SPARC32-LABEL: test_lrint_ixx_f64:
; SPARC32:         call lrint
;
; SPARC64-LABEL: test_lrint_ixx_f64:
; SPARC64:         call lrint
  %res = tail call ITy @llvm.lrint.ITy.f64(double %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f64(double %x) nounwind {
; SPARC32-LABEL: test_llrint_ixx_f64:
; SPARC32:         call llrint
;
; SPARC64-LABEL: test_llrint_ixx_f64:
; SPARC64:         call llrint
  %res = tail call ITy @llvm.llrint.ITy.f64(double %x)
  ret ITy %res
}

; FIXME(#41838): unsupported type
; define ITy @test_lrint_ixx_f128(fp128 %x) nounwind {
;   %res = tail call ITy @llvm.lrint.ITy.f128(fp128 %x)
;   ret ITy %res
; }

; define ITy @test_llrint_ixx_f128(fp128 %x) nounwind {
;   %res = tail call ITy @llvm.llrint.ITy.f128(fp128 %x)
;   ret ITy %res
; }

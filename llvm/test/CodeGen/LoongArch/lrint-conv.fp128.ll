; fp128 lrint/llrint on loongarch64, where long double is fp128 and the call is
; the correct-width libcall (unlike loongarch32, see lrint-conv.ll).

; RUN: sed 's/ITy/i32/g' %s | llc -mtriple=loongarch64 | FileCheck %s --check-prefixes=LA64-I32
; RUN: sed 's/ITy/i64/g' %s | llc -mtriple=loongarch64 | FileCheck %s --check-prefixes=LA64-I64

define ITy @test_lrint_ixx_f128(fp128 %x) nounwind {
; LA64-I32-LABEL: test_lrint_ixx_f128:
; LA64-I32:         pcaddu18i $ra, %call36(lrintl)
;
; LA64-I64-LABEL: test_lrint_ixx_f128:
; LA64-I64:         pcaddu18i $ra, %call36(lrintl)
  %res = tail call ITy @llvm.lrint.ITy.f128(fp128 %x)
  ret ITy %res
}

define ITy @test_llrint_ixx_f128(fp128 %x) nounwind {
; LA64-I32-LABEL: test_llrint_ixx_f128:
; LA64-I32:         pcaddu18i $ra, %call36(llrintl)
;
; LA64-I64-LABEL: test_llrint_ixx_f128:
; LA64-I64:         pcaddu18i $ra, %call36(llrintl)
  %res = tail call ITy @llvm.llrint.ITy.f128(fp128 %x)
  ret ITy %res
}

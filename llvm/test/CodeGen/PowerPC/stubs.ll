; RUN: llc %s -o - -mtriple=powerpc-apple-darwin8 | FileCheck %s
define ppc_fp128 @test1(i64 %X) nounwind readnone {
entry:
  %0 = sitofp i64 %X to ppc_fp128
  ret ppc_fp128 %0
}

; CHECK: _test1:
; CHECK: bl ___floatditf$stub
; CHECK: 	.section	__TEXT,__symbol_stub1,symbol_stubs,pure_instructions,16
; CHECK: ___floatditf$stub:
; CHECK: 	.indirect_symbol ___floatditf
; CHECK: 	lis r11,ha16(___floatditf$lazy_ptr)
; CHECK: 	lwzu r12,lo16(___floatditf$lazy_ptr)(r11)
; CHECK: 	mtctr r12
; CHECK: 	bctr
; CHECK: 	.section	__DATA,__la_symbol_ptr,lazy_symbol_pointers
; CHECK: ___floatditf$lazy_ptr:
; CHECK: 	.indirect_symbol ___floatditf
; CHECK: 	.long dyld_stub_binding_helper



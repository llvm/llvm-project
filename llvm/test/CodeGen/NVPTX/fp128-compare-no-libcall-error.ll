; RUN: not llc -mtriple=nvptx64 -filetype=null %s 2>&1 | FileCheck %s

; NVPTX has no fp128 compare libcall, so this should diagnose a missing libcall
; rather than crash. The reported compare kind is the original one, not the
; inverted or rewritten form used to select the libcall.

; CHECK: error: no libcall available to soften floating-point setoeq compare with type f128
define i1 @test_fcmp_oeq_f128(fp128 %a, fp128 %b) {
  %r = fcmp oeq fp128 %a, %b
  ret i1 %r
}

; CHECK: error: no libcall available to soften floating-point setult compare with type f128
define i1 @test_fcmp_ult_f128(fp128 %a, fp128 %b) {
  %r = fcmp ult fp128 %a, %b
  ret i1 %r
}

; CHECK: error: no libcall available to soften floating-point setone compare with type f128
define i1 @test_fcmp_one_f128(fp128 %a, fp128 %b) {
  %r = fcmp one fp128 %a, %b
  ret i1 %r
}

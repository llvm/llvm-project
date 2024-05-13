; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; PR11319

@i8_res  = global <2 x i8> <i8 0, i8 0>
@i8_src1 = global <2 x i8> <i8 1, i8 2>
@i8_src2 = global <2 x i8> <i8 2, i8 1>

define void @test_neon_vector_add_2xi8() nounwind {
; CHECK-LABEL: test_neon_vector_add_2xi8:
  %1 = load <2 x i8>, ptr @i8_src1
  %2 = load <2 x i8>, ptr @i8_src2
  %3 = add <2 x i8> %1, %2
  store <2 x i8> %3, ptr @i8_res
  ret void
}

define void @test_neon_ld_st_volatile_with_ashr_2xi8() {
; CHECK-LABEL: test_neon_ld_st_volatile_with_ashr_2xi8:
  %1 = load volatile <2 x i8>, ptr @i8_src1
  %2 = load volatile <2 x i8>, ptr @i8_src2
  %3 = ashr <2 x i8> %1, %2
  store volatile <2 x i8> %3, ptr @i8_res
  ret void
}

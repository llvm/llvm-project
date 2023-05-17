; RUN: llc -verify-machineinstrs -mtriple=armv7-eabi -mattr=+neon %s -o - | FileCheck %s --check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs -mtriple=armv7eb-eabi -mattr=+neon %s -o - | FileCheck %s --check-prefix=CHECK-BE

define void @ld_st_vec_i8(ptr %A, ptr %B) nounwind {
;CHECK-LE-LABEL: ld_st_vec_i8:
;CHECK-LE: vld1.8 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-LE-NOT: vrev
;CHECK-LE: vst1.8 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

;CHECK-BE-LABEL: ld_st_vec_i8:
;CHECK-BE: vld1.8 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-BE: vrev64.8 [[Q1:q[0-9]+]], [[Q2:q[0-9]+]]
;CHECK-BE: vrev64.8 [[Q1]], [[Q2]]
;CHECK-BE: vst1.8 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

%load = load <16 x i8>, ptr %A, align 1
store <16 x i8> %load, ptr %B, align 1
ret void
}

define void @ld_st_vec_i16(ptr %A, ptr %B) nounwind {
;CHECK-LE-LABEL: ld_st_vec_i16:
;CHECK-LE: vld1.16 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-LE-NOT: vrev
;CHECK-LE: vst1.16 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

;CHECK-BE-LABEL: ld_st_vec_i16:
;CHECK-BE: vld1.16 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-BE: vrev64.16 [[Q1:q[0-9]+]], [[Q2:q[0-9]+]]
;CHECK-BE: vrev64.16 [[Q1]], [[Q2]]
;CHECK-BE: vst1.16 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

%load = load <8 x i16>, ptr %A, align 2
store <8 x i16> %load, ptr %B, align 2
ret void
}

define void @ld_st_vec_i32(ptr %A, ptr %B) nounwind {
;CHECK-LE-LABEL: ld_st_vec_i32:
;CHECK-LE: vld1.32 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-LE-NOT: vrev
;CHECK-LE: vst1.32 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

;CHECK-BE-LABEL: ld_st_vec_i32:
;CHECK-BE: vldmia {{r[0-9]+}}, {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}
;CHECK-BE-NOT: vrev
;CHECK-BE: vstmia {{r[0-9]+}}, {[[D1]], [[D2]]}

%load = load <4 x i32>, ptr %A, align 4
store <4 x i32> %load, ptr %B, align 4
ret void
}

define void @ld_st_vec_double(ptr %A, ptr %B) nounwind {
;CHECK-LE-LABEL: ld_st_vec_double:
;CHECK-LE: vld1.64 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-LE-NOT: vrev
;CHECK-LE: vst1.64 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

;CHECK-BE-LABEL: ld_st_vec_double:
;CHECK-BE: vld1.64 {[[D1:d[0-9]+]], [[D2:d[0-9]+]]}, [{{r[0-9]+}}]
;CHECK-BE-NOT: vrev
;CHECK-BE: vst1.64 {[[D1]], [[D2]]}, [{{r[0-9]+}}]

%load = load <2 x double>, ptr %A, align 8
store <2 x double> %load, ptr %B, align 8
ret void
}

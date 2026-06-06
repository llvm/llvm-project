; RUN: llvm-as < %s | llvm-dis | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }

declare %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2.v8i8.p0i8(i8*)

; CHECK-LABEL: define %struct.__neon_int8x8x2_t @test_named_struct_return(ptr %A) {
; CHECK:  %1 = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr align 16 %A)
; CHECK:  %2 = extractvalue { <8 x i8>, <8 x i8> } %1, 0
; CHECK:  %3 = insertvalue %struct.__neon_int8x8x2_t poison, <8 x i8> %2, 0
; CHECK:  %4 = extractvalue { <8 x i8>, <8 x i8> } %1, 1
; CHECK:  %5 = insertvalue %struct.__neon_int8x8x2_t %3, <8 x i8> %4, 1
; CHECK:  ret %struct.__neon_int8x8x2_t %5

define %struct.__neon_int8x8x2_t @test_named_struct_return(ptr %A) {
  %val = call %struct.__neon_int8x8x2_t @llvm.aarch64.neon.ld2.v8i8.p0i8(ptr align 16 %A)
  ret %struct.__neon_int8x8x2_t %val
}

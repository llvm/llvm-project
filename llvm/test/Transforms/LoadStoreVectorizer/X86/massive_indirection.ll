; RUN: opt %s -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -mcpu=skx -S -o %t.out.ll
; RUN: FileCheck -input-file=%t.out.ll %s

; This test verifies that the vectorizer can handle an extended sequence of
; getelementptr instructions and generate longer vectors. With special handling,
; some elements can still be vectorized even if they require looking up the
; common underlying object deeper than 6 levels from the original pointer.

; The test below is the simplified version of actual performance oriented
; workload; the offsets in getelementptr instructions are similar or same for
; the test simplicity.

define void @v1_v2_v4_v1_to_v8_levels_6_7_8_8(i32 %arg0, ptr align 16 %arg1) {
; CHECK-LABEL: @v1_v2_v4_v1_to_v8_levels_6_7_8_8
; CHECK: store <8 x half>

  %level1 = getelementptr inbounds i8, ptr %arg1, i32 917504
  %level2 = getelementptr i8, ptr %level1, i32 %arg0
  %level3 = getelementptr i8, ptr %level2, i32 32768
  %level4 = getelementptr inbounds i8, ptr %level3, i32 %arg0
  %level5 = getelementptr i8, ptr %level4, i32 %arg0

  %a6 = getelementptr i8, ptr %level5, i32 %arg0
  %b7 = getelementptr i8, ptr %a6, i32 2
  %c8 = getelementptr i8, ptr %b7, i32 8
  %d8 = getelementptr inbounds i8, ptr %b7, i32 12

  store half 0xH0000, ptr %a6, align 16
  store <4 x half> zeroinitializer, ptr %b7, align 2
  store <2 x half> zeroinitializer, ptr %c8, align 2
  store half 0xH0000, ptr %d8, align 2
  ret void
}

define void @v1x8_levels_6_7_8_9_10_11_12_13(i32 %arg0, ptr align 16 %arg1) {
; CHECK-LABEL: @v1x8_levels_6_7_8_9_10_11_12_13
; CHECK: store <8 x half>

  %level1 = getelementptr inbounds i8, ptr %arg1, i32 917504
  %level2 = getelementptr i8, ptr %level1, i32 %arg0
  %level3 = getelementptr i8, ptr %level2, i32 32768
  %level4 = getelementptr inbounds i8, ptr %level3, i32 %arg0
  %level5 = getelementptr i8, ptr %level4, i32 %arg0

  %a6 = getelementptr i8, ptr %level5, i32 %arg0
  %b7 = getelementptr i8, ptr %a6, i32 2
  %c8 = getelementptr i8, ptr %b7, i32 2
  %d9 = getelementptr inbounds i8, ptr %c8, i32 2
  %e10 = getelementptr inbounds i8, ptr %d9, i32 2
  %f11 = getelementptr inbounds i8, ptr %e10, i32 2
  %g12 = getelementptr inbounds i8, ptr %f11, i32 2
  %h13 = getelementptr inbounds i8, ptr %g12, i32 2

  store half 0xH0000, ptr %a6, align 16
  store half 0xH0000, ptr %b7, align 2
  store half 0xH0000, ptr %c8, align 2
  store half 0xH0000, ptr %d9, align 2
  store half 0xH0000, ptr %e10, align 8
  store half 0xH0000, ptr %f11, align 2
  store half 0xH0000, ptr %g12, align 2
  store half 0xH0000, ptr %h13, align 2
  ret void
}

; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; Verify that a constant with size in excess of 32-bit SIZE_MAX doesn't
; cause trouble.  This test exercises an internal limit set arbitrarily
; at 64K for the largest supported zeroinitiazer. If the limit changes
; the test might need to be adjusted.

declare i8* @memrchr(i8*, i32, i64)

@a = constant <{ i8, [4294967295 x i8] }> <{ i8 1, [4294967295 x i8] zeroinitializer }>

; Verify reading an initializer INT32_MAX + 1 bytes large (starting at
; offset 2147483647 into a which is UINT32_MAX bytes in size).

define i8* @call_a_pi32max_p1() {
; CHECK-LABEL: @call_a_pi32max_p1(
; CHECK-NEXT:    [[CHR:%.*]] = tail call i8* @memrchr(i8* noundef nonnull dereferenceable(2147483647) getelementptr inbounds (<{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i64 0, i32 1, i64 2147483647), i32 0, i64 2147483647)
; CHECK-NEXT:    ret i8* [[CHR]]
;
  %ptr = getelementptr <{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i32 0, i32 1, i32 2147483647
  %chr = tail call i8* @memrchr(i8* %ptr, i32 0, i64 2147483647)
  ret i8* %chr
}

; Verify reading an initializer INT32_MAX bytes large (starting at offset
; 2147483648 into a which is UINT32_MAX bytes in size).

define i8* @call_a_pi32max() {
; CHECK-LABEL: @call_a_pi32max(
; CHECK-NEXT:    [[CHR:%.*]] = tail call i8* @memrchr(i8* noundef nonnull dereferenceable(2147483647) getelementptr inbounds (<{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i64 0, i32 1, i64 2147483648), i32 0, i64 2147483647)
; CHECK-NEXT:    ret i8* [[CHR]]
;
  %ptr = getelementptr <{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i32 0, i32 1, i64 2147483648
  %chr = tail call i8* @memrchr(i8* %ptr, i32 0, i64 2147483647)
  ret i8* %chr
}


; Verify reading an initializer UINT32_MAX bytes large (starting at offset
; 1 into a).

define i8* @call_a_pui32max() {
; CHECK-LABEL: @call_a_pui32max(
; CHECK-NEXT:    [[CHR:%.*]] = tail call i8* @memrchr(i8* noundef nonnull dereferenceable(4294967295) getelementptr inbounds (<{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i64 0, i32 1, i64 0), i32 0, i64 4294967295)
; CHECK-NEXT:    ret i8* [[CHR]]
;
  %ptr = getelementptr <{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i32 0, i32 1, i32 0
  %chr = tail call i8* @memrchr(i8* %ptr, i32 0, i64 4294967295)
  ret i8* %chr
}

; Verify reading an initializer UINT32_MAX + 1 bytes large (all of a).

define i8* @call_a_puimax_p1() {
; CHECK-LABEL: @call_a_puimax_p1(
; CHECK-NEXT:    [[CHR:%.*]] = tail call i8* @memrchr(i8* noundef nonnull dereferenceable(4294967296) getelementptr inbounds (<{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i64 0, i32 0), i32 0, i64 4294967296)
; CHECK-NEXT:    ret i8* [[CHR]]
;
  %ptr = getelementptr <{ i8, [4294967295 x i8] }>, <{ i8, [4294967295 x i8] }>* @a, i32 0, i32 0
  %chr = tail call i8* @memrchr(i8* %ptr, i32 0, i64 4294967296)
  ret i8* %chr
}
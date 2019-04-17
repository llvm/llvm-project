; RUN: opt -S -codegenprepare -mtriple=arm64-apple-ios7.0 %s | FileCheck %s

%foo = type { i8 }

define %foo @test_merge(i32 %in) {
; CHECK-LABEL: @test_merge

  ; CodeGenPrepare was requesting the EVT for { i8 } to determine
  ; whether the insertvalue user of the trunc was legal. This
  ; asserted.

; CHECK: insertvalue %foo undef, i8 %byte, 0
  %lobit = lshr i32 %in, 31
  %byte = trunc i32 %lobit to i8
  %struct = insertvalue %foo undef, i8 %byte, 0
  ret %"foo" %struct
}

define i64* @test_merge_PR21548(i32 %a, i64* %p1, i64* %p2, i64* %p3) {
; CHECK-LABEL: @test_merge_PR21548
  %as = lshr i32 %a, 3
  %Tr = trunc i32 %as to i1
  br i1 %Tr, label %BB2, label %BB3

BB2:
  ; Similarly to above:
  ; CodeGenPrepare was requesting the EVT for i8* to determine
  ; whether the select user of the trunc was legal. This asserted.

; CHECK: select i1 {{%.*}}, i64* %p1, i64* %p2
  %p = select i1 %Tr, i64* %p1, i64* %p2
  ret i64* %p

BB3:
  ret i64* %p3
}

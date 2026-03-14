; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s

declare i32 @ret32(i32 returned)
declare i64 @ret64(i64 returned)

define i64 @test1(i64 %val) {
; CHECK-LABEL: test1:
; CHECK-NOT: jmp
; CHECK: callq
  %in = trunc i64 %val to i32
  tail call i32 @ret32(i32 returned %in)
  ret i64 %val
}

define i32 @test2(i64 %val) {
; CHECK-LABEL: test2:
; CHECK: jmp
; CHECK-NOT: callq
  %in = trunc i64 %val to i32
  tail call i64 @ret64(i64 returned %val)
  ret i32 %in
}

define i32 @test3(i64 %in) {
; CHECK-LABEL: test3:
; CHECK: jmp
; CHECK-NOT: callq
  %small = trunc i64 %in to i32
  tail call i32 @ret32(i32 returned %small)
  ret i32 %small
}

declare {i32, i8} @take_i32_i8({i32, i8} returned)
define { i8, i8 } @test_nocommon_value({i32, i32} %in) {
; CHECK-LABEL: test_nocommon_value
; CHECK: jmp

  %first = extractvalue {i32, i32} %in, 0
  %first.trunc = trunc i32 %first to i8

  %second = extractvalue {i32, i32} %in, 1
  %second.trunc = trunc i32 %second to i8

  %tmp = insertvalue {i32, i8} undef, i32 %first, 0
  %callval = insertvalue {i32, i8} %tmp, i8 %second.trunc, 1
  tail call {i32, i8} @take_i32_i8({i32, i8} returned %callval)

  %restmp = insertvalue {i8, i8} undef, i8 %first.trunc, 0
  %res = insertvalue {i8, i8} %restmp, i8 %second.trunc, 1
  ret {i8, i8} %res
}

declare {i32, {i32, i32}} @give_i32_i32_i32()
define {{i32, i32}, i32} @test_structs_different_shape() {
; CHECK-LABEL: test_structs_different_shape
; CHECK: jmp
  %val = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %first = extractvalue {i32, {i32, i32}} %val, 0
  %second = extractvalue {i32, {i32, i32}} %val, 1, 0
  %third = extractvalue {i32, {i32, i32}} %val, 1, 1

  %restmp = insertvalue {{i32, i32}, i32} undef, i32 %first, 0, 0
  %reseventmper = insertvalue {{i32, i32}, i32} %restmp, i32 %second, 0, 1
  %res = insertvalue {{i32, i32}, i32} %reseventmper, i32 %third, 1

  ret {{i32, i32}, i32} %res
}

define i64 @test_undef_asymmetry() {
; CHECK: test_undef_asymmetry
; CHECK-NOT: jmp
  tail call i64 @ret64(i64 returned undef)
  ret i64 2
}

define {{}, {{}, i32, {}}, [1 x i32]} @evil_empty_aggregates() {
; CHECK-LABEL: evil_empty_aggregates
; CHECK: jmp
  %agg = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %first = extractvalue {i32, {i32, i32}} %agg, 0
  %second = extractvalue {i32, {i32, i32}} %agg, 1, 0

  %restmp = insertvalue {{}, {{}, i32, {}}, [1 x i32]} undef, i32 %first, 1, 1
  %res = insertvalue {{}, {{}, i32, {}}, [1 x i32]} %restmp, i32 %second, 2, 0
  ret {{}, {{}, i32, {}}, [1 x i32]} %res
}

define i32 @structure_is_unimportant() {
; CHECK-LABEL: structure_is_unimportant
; CHECK: jmp
  %val = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %res = extractvalue {i32, {i32, i32}} %val, 0
  ret i32 %res
}

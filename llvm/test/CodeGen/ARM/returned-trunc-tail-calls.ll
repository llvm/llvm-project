; RUN: llc < %s -mtriple=armv7 | FileCheck %s

declare i16 @ret16(i16 returned)
declare i32 @ret32(i32 returned)

define i32 @test1(i32 %val) {
; CHECK-LABEL: test1:
; CHECK: bl {{_?}}ret16
  %in = trunc i32 %val to i16
  tail call i16 @ret16(i16 returned %in)
  ret i32 %val
}

define i16 @test2(i32 %val) {
; CHECK-LABEL: test2:
; CHECK: b {{_?}}ret16
  %in = trunc i32 %val to i16
  tail call i16 @ret16(i16 returned %in)
  ret i16 %in
}

declare {i32, i8} @take_i32_i8({i32, i8} returned)
define { i8, i8 } @test_nocommon_value({i32, i32} %in) {
; CHECK-LABEL: test_nocommon_value:
; CHECK: b {{_?}}take_i32_i8

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
; CHECK-LABEL: test_structs_different_shape:
; CHECK: b {{_?}}give_i32_i32_i32
  %val = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %first = extractvalue {i32, {i32, i32}} %val, 0
  %second = extractvalue {i32, {i32, i32}} %val, 1, 0
  %third = extractvalue {i32, {i32, i32}} %val, 1, 1

  %restmp = insertvalue {{i32, i32}, i32} undef, i32 %first, 0, 0
  %reseventmper = insertvalue {{i32, i32}, i32} %restmp, i32 %second, 0, 1
  %res = insertvalue {{i32, i32}, i32} %reseventmper, i32 %third, 1

  ret {{i32, i32}, i32} %res
}

define i32 @test_undef_asymmetry() {
; CHECK: test_undef_asymmetry:
; CHECK: bl {{_?}}ret32
; CHECK-NOT: jmp
  tail call i32 @ret32(i32 returned undef)
  ret i32 2
}

define {{}, {{}, i32, {}}, [1 x i32]} @evil_empty_aggregates() {
; CHECK-LABEL: evil_empty_aggregates:
; CHECK: b {{_?}}give_i32_i32_i32
  %agg = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %first = extractvalue {i32, {i32, i32}} %agg, 0
  %second = extractvalue {i32, {i32, i32}} %agg, 1, 0

  %restmp = insertvalue {{}, {{}, i32, {}}, [1 x i32]} undef, i32 %first, 1, 1
  %res = insertvalue {{}, {{}, i32, {}}, [1 x i32]} %restmp, i32 %second, 2, 0
  ret {{}, {{}, i32, {}}, [1 x i32]} %res
}

define i32 @structure_is_unimportant() {
; CHECK-LABEL: structure_is_unimportant:
; CHECK: b {{_?}}give_i32_i32_i32
  %val = tail call {i32, {i32, i32}} @give_i32_i32_i32()

  %res = extractvalue {i32, {i32, i32}} %val, 0
  ret i32 %res
}

declare i64 @give_i64()
define i64 @direct_i64_ok() {
; CHECK-LABEL: direct_i64_ok:
; CHECK: b {{_?}}give_i64
  %val = tail call i64 @give_i64()
  ret i64 %val
}

declare {i64, i32} @give_i64_i32()
define {i32, i32} @trunc_i64_not_ok() {
; CHECK-LABEL: trunc_i64_not_ok:
; CHECK: bl {{_?}}give_i64_i32
  %agg = tail call {i64, i32} @give_i64_i32()

  %first = extractvalue {i64, i32} %agg, 0
  %second = extractvalue {i64, i32} %agg, 1
  %first.trunc = trunc i64 %first to i32

  %tmp = insertvalue {i32, i32} undef, i32 %first.trunc, 0
  %ret = insertvalue {i32, i32} %tmp, i32 %second, 1

  ret {i32, i32} %ret
}

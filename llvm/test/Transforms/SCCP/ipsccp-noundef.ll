; RUN: opt -S -passes=ipsccp < %s | FileCheck %s
@g = external global i8

define internal noundef i32 @ret_noundef() {
; CHECK-LABEL: define internal noundef i32 @ret_noundef() {
; CHECK-NEXT:    ret i32 undef
;
  ret i32 0
}

define internal dereferenceable(1) ptr @ret_dereferenceable() {
; CHECK-LABEL: define internal dereferenceable(1) ptr @ret_dereferenceable() {
; CHECK-NEXT:    ret ptr undef
;
  ret ptr @g
}

define internal dereferenceable_or_null(1) ptr @ret_dereferenceable_or_null() {
; CHECK-LABEL: define internal dereferenceable_or_null(1) ptr @ret_dereferenceable_or_null() {
; CHECK-NEXT:    ret ptr undef
;
  ret ptr @g
}

define internal nonnull ptr @ret_nonnull() {
; CHECK-LABEL: define internal nonnull ptr @ret_nonnull() {
; CHECK-NEXT:    ret ptr undef
;
  ret ptr @g
}

define internal nonnull noundef ptr @ret_nonnull_noundef() {
; CHECK-LABEL: define internal noundef nonnull ptr @ret_nonnull_noundef() {
; CHECK-NEXT:    ret ptr undef
;
  ret ptr @g
}

define void @test() {
; CHECK-LABEL: define void @test() {
; CHECK-NEXT:    [[TMP1:%.*]] = call noundef i32 @ret_noundef()
; CHECK-NEXT:    [[TMP2:%.*]] = call dereferenceable(1) ptr @ret_dereferenceable()
; CHECK-NEXT:    [[TMP3:%.*]] = call dereferenceable_or_null(1) ptr @ret_dereferenceable_or_null()
; CHECK-NEXT:    [[TMP4:%.*]] = call nonnull ptr @ret_nonnull()
; CHECK-NEXT:    [[TMP5:%.*]] = call noundef nonnull ptr @ret_nonnull_noundef()
; CHECK-NEXT:    ret void
;
  call noundef i32 @ret_noundef()
  call dereferenceable(1) ptr @ret_dereferenceable()
  call dereferenceable_or_null(1) ptr @ret_dereferenceable_or_null()
  call nonnull ptr @ret_nonnull()
  call nonnull noundef ptr @ret_nonnull_noundef()
  ret void
}

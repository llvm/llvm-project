; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes='require<profile-summary>,codegenprepare' -S < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t.o
;
; CodeGenPrepare must not rewrite fortified libcalls if doing so would drop
; musttail or notail. Calls without those constraints should still be lowered.

declare ptr @__strcat_chk(ptr, ptr, i64)
declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

define ptr @musttail_strcat(ptr %dst, ptr %src, i64 %unused) {
; CHECK-LABEL: define ptr @musttail_strcat(
; CHECK:       musttail call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
; CHECK-NEXT:  ret ptr
  %ret = musttail call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
  ret ptr %ret
}

define ptr @musttail_memcpy(ptr %dst, ptr %src, i64 %len, i64 %unused) {
; CHECK-LABEL: define ptr @musttail_memcpy(
; CHECK:       musttail call ptr @__memcpy_chk(ptr %dst, ptr %src, i64 %len, i64 -1)
; CHECK-NEXT:  ret ptr
  %ret = musttail call ptr @__memcpy_chk(ptr %dst, ptr %src, i64 %len, i64 -1)
  ret ptr %ret
}

define ptr @notail_strcat(ptr %dst, ptr %src) {
; CHECK-LABEL: define ptr @notail_strcat(
; CHECK:       notail call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
  %ret = notail call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
  ret ptr %ret
}

define ptr @ordinary_strcat(ptr %dst, ptr %src) {
; CHECK-LABEL: define ptr @ordinary_strcat(
; CHECK-NOT:   @__strcat_chk
; CHECK:       call ptr @strcat(ptr %dst, ptr %src)
  %ret = call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
  ret ptr %ret
}

define ptr @notail_memcpy(ptr %dst, ptr %src, i64 %len) {
; CHECK-LABEL: define ptr @notail_memcpy(
; CHECK:       notail call ptr @__memcpy_chk(ptr %dst, ptr %src, i64 %len, i64 -1)
; CHECK-NEXT:  ret ptr
  %ret = notail call ptr @__memcpy_chk(ptr %dst, ptr %src, i64 %len, i64 -1)
  ret ptr %ret
}

define ptr @tail_strcat(ptr %dst, ptr %src) {
; CHECK-LABEL: define ptr @tail_strcat(
; CHECK-NOT:   @__strcat_chk
; CHECK:       tail call ptr @strcat(ptr %dst, ptr %src)
  %ret = tail call ptr @__strcat_chk(ptr %dst, ptr %src, i64 -1)
  ret ptr %ret
}

define ptr @ordinary_memcpy(ptr %dst, ptr %src, i64 %len) {
; CHECK-LABEL: define ptr @ordinary_memcpy(
; CHECK-NOT:   @__memcpy_chk
; CHECK:       call void @llvm.memcpy
; CHECK-NEXT:  ret ptr %dst
  %ret = call ptr @__memcpy_chk(ptr %dst, ptr %src, i64 %len, i64 -1)
  ret ptr %ret
}

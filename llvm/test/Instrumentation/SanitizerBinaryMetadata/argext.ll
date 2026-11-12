; RUN: opt < %s -mtriple=s390x-linux-gnu -passes='module(sanmd-module)' \
; RUN:   -sanitizer-metadata-atomics -S | FileCheck %s
; REQUIRES: systemz-registered-target

; Check that the i32 version argument of the metadata init functions carries
; zeroext.

; CHECK-DAG: declare {{.*}}void @__sanitizer_metadata_atomics_add(i32 zeroext, ptr, ptr)
; CHECK-DAG: declare {{.*}}void @__sanitizer_metadata_atomics_del(i32 zeroext, ptr, ptr)
; CHECK-DAG: declare {{.*}}void @__sanitizer_metadata_covered_add(i32 zeroext, ptr, ptr)
; CHECK-DAG: declare {{.*}}void @__sanitizer_metadata_covered_del(i32 zeroext, ptr, ptr)

define i8 @foo(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a unordered, align 1
  ret i8 %0
}

; RUN: opt -S -passes=attributor %s | FileCheck %s --check-prefixes=ATTR
; RUN: opt -S -passes='attributor,function(gvn)' %s | FileCheck %s --check-prefixes=GVN

; Regression for #243743: a pointer returned by a callee and overwritten in the
; caller must not receive !invariant.load. GVN must not fold the icmp to true.

define ptr @get(ptr %t) {
; ATTR-LABEL: define {{.*}}@get(
; ATTR: ret ptr
  %b = load ptr, ptr %t
  ret ptr %b
}

define i1 @memcpy_via_returned_pointer(ptr %t, ptr %src) {
; ATTR-LABEL: define {{.*}}@memcpy_via_returned_pointer(
; ATTR: %[[P:.*]] = call ptr @get(
; ATTR: %[[G:.*]] = getelementptr inbounds i8, ptr %[[P]], i64 8
; ATTR: %[[OLD:.*]] = load i64, ptr %[[G]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: call void @llvm.memcpy
; ATTR: %[[NEW:.*]] = load i64, ptr %[[G]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: %[[SAME:.*]] = icmp eq i64 %[[OLD]], %[[NEW]]
; ATTR: ret i1 %[[SAME]]
;
; GVN-LABEL: define {{.*}}@memcpy_via_returned_pointer(
; GVN-NOT: ret i1 true
; GVN: %[[SAME:.*]] = icmp eq i64
; GVN: ret i1 %[[SAME]]
;
  %p = call ptr @get(ptr %t)
  %g = getelementptr inbounds i8, ptr %p, i64 8
  %old = load i64, ptr %g
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %src, i64 16, i1 false)
  %new = load i64, ptr %g
  %same = icmp eq i64 %old, %new
  ret i1 %same
}

define i1 @store_via_returned_pointer(ptr %t, i64 %v) {
; ATTR-LABEL: define {{.*}}@store_via_returned_pointer(
; ATTR: %[[P:.*]] = call ptr @get(
; ATTR: %[[G:.*]] = getelementptr inbounds i8, ptr %[[P]], i64 8
; ATTR: %[[OLD:.*]] = load i64, ptr %[[G]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: store i64
; ATTR: %[[NEW:.*]] = load i64, ptr %[[G]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: %[[SAME:.*]] = icmp eq i64 %[[OLD]], %[[NEW]]
; ATTR: ret i1 %[[SAME]]
;
; GVN-LABEL: define {{.*}}@store_via_returned_pointer(
; GVN-NOT: ret i1 true
; GVN: %[[SAME:.*]] = icmp eq i64
; GVN: ret i1 %[[SAME]]
;
  %p = call ptr @get(ptr %t)
  %g = getelementptr inbounds i8, ptr %p, i64 8
  %old = load i64, ptr %g
  %g2 = getelementptr inbounds i8, ptr %p, i64 8
  store i64 %v, ptr %g2
  %new = load i64, ptr %g
  %same = icmp eq i64 %old, %new
  ret i1 %same
}

; Use-local readonly on %q does not prove !invariant.load if %alias may write
; the same bytes.
define i32 @alias_write_through_other_pointer(ptr %p, ptr %alias) {
; ATTR-LABEL: define {{.*}}@alias_write_through_other_pointer(
; ATTR: %[[Q:.*]] = load ptr, ptr {{.*}}, align {{[0-9]+}}{{$}}
; ATTR: store i32 1, ptr
; ATTR: %[[W:.*]] = load i32, ptr %[[Q]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: ret i32 %[[W]]
;
  %q = load ptr, ptr %p
  %v = load i32, ptr %q
  store i32 1, ptr %alias
  %w = load i32, ptr %q
  ret i32 %w
}

; Returning a GEP of a noalias argument must not tag loads through it as
; invariant: the callee exports the pointer for the caller to mutate.
define ptr @returned_gep(ptr noalias %p, ptr %out) {
; ATTR-LABEL: define {{.*}}@returned_gep(
; ATTR: %[[G:.*]] = getelementptr
; ATTR: %[[V:.*]] = load i64, ptr %[[G]], align {{[0-9]+}}{{$}}
; ATTR-NOT: !invariant.load
; ATTR: store i64 %[[V]], ptr
; ATTR: ret ptr %[[G]]
;
  %g = getelementptr i8, ptr %p, i64 8
  %v = load i64, ptr %g
  store i64 %v, ptr %out
  ret ptr %g
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

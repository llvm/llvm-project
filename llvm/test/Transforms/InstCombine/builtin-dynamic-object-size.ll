; RUN: opt -instcombine -S < %s | FileCheck %s --dump-input-on-failure

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Function Attrs: nounwind ssp uwtable
define i64 @weird_identity_but_ok(i64 %sz) {
entry:
  %call = tail call i8* @malloc(i64 %sz)
  %calc_size = tail call i64 @llvm.objectsize.i64.p0i8(i8* %call, i1 false, i1 true, i1 true)
  tail call void @free(i8* %call)
  ret i64 %calc_size
}

; CHECK:      define i64 @weird_identity_but_ok(i64 %sz)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i64 %sz
; CHECK-NEXT: }

define i64 @phis_are_neat(i1 %which) {
entry:
  br i1 %which, label %first_label, label %second_label

first_label:
  %first_call = call i8* @malloc(i64 10)
  br label %join_label

second_label:
  %second_call = call i8* @malloc(i64 30)
  br label %join_label

join_label:
  %joined = phi i8* [ %first_call, %first_label ], [ %second_call, %second_label ]
  %calc_size = tail call i64 @llvm.objectsize.i64.p0i8(i8* %joined, i1 false, i1 true, i1 true)
  ret i64 %calc_size
}

; CHECK:      %0 = phi i64 [ 10, %first_label ], [ 30, %second_label ]
; CHECK-NEXT: ret i64 %0

define i64 @internal_pointer(i64 %sz) {
entry:
  %ptr = call i8* @malloc(i64 %sz)
  %ptr2 = getelementptr inbounds i8, i8* %ptr, i32 2
  %calc_size = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr2, i1 false, i1 true, i1 true)
  ret i64 %calc_size
}

; CHECK:      define i64 @internal_pointer(i64 %sz)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %sz, -2
; CHECK-NEXT:   %1 = icmp ult i64 %sz, 2
; CHECK-NEXT:   %2 = select i1 %1, i64 0, i64 %0
; CHECK-NEXT:   ret i64 %2
; CHECK-NEXT: }

define i64 @uses_nullptr_no_fold() {
entry:
  %res = call i64 @llvm.objectsize.i64.p0i8(i8* null, i1 false, i1 true, i1 true)
  ret i64 %res
}

; CHECK: %res = call i64 @llvm.objectsize.i64.p0i8(i8* null, i1 false, i1 true, i1 true)

define i64 @uses_nullptr_fold() {
entry:
  ; NOTE: the third parameter to this call is false, unlike above.
  %res = call i64 @llvm.objectsize.i64.p0i8(i8* null, i1 false, i1 false, i1 true)
  ret i64 %res
}

; CHECK: ret i64 0

; Function Attrs: nounwind allocsize(0)
declare i8* @malloc(i64)

declare i8* @get_unknown_buffer()

; Function Attrs: nounwind
declare void @free(i8* nocapture)

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1, i1)

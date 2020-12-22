; RUN: opt -instcombine -S < %s | FileCheck %s

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
; CHECK:   ret i64 %sz
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
; CHECK-NEXT:   %0 = call i64 @llvm.usub.sat.i64(i64 %sz, i64 2)
; CHECK-NEXT:   ret i64 %0
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

@d = common global i8 0, align 1
@c = common global i32 0, align 4

; Function Attrs: nounwind
define void @f() {
entry:
  %.pr = load i32, i32* @c, align 4
  %tobool4 = icmp eq i32 %.pr, 0
  br i1 %tobool4, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %dp.05 = phi i8* [ %add.ptr, %for.body ], [ @d, %entry ]
  %0 = tail call i64 @llvm.objectsize.i64.p0i8(i8* %dp.05, i1 false, i1 true, i1 true)
  %conv = trunc i64 %0 to i32
  tail call void @bury(i32 %conv) #3
  %1 = load i32, i32* @c, align 4
  %idx.ext = sext i32 %1 to i64
  %add.ptr.offs = add i64 %idx.ext, 0
  %2 = add i64 undef, %add.ptr.offs
  %add.ptr = getelementptr inbounds i8, i8* %dp.05, i64 %idx.ext
  %add = shl nsw i32 %1, 1
  store i32 %add, i32* @c, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK:   define void @f()
; CHECK:     call i64 @llvm.objectsize.i64.p0i8(

define void @bdos_cmpm1(i64 %alloc) {
entry:
  %obj = call i8* @malloc(i64 %alloc)
  %objsize = call i64 @llvm.objectsize.i64.p0i8(i8* %obj, i1 0, i1 0, i1 1)
  %cmp.not = icmp eq i64 %objsize, -1
  br i1 %cmp.not, label %if.else, label %if.then

if.then:
  call void @fortified_chk(i8* %obj, i64 %objsize)
  br label %if.end

if.else:
  call void @unfortified(i8* %obj, i64 %objsize)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK:  define void @bdos_cmpm1(
; CHECK:    [[TMP:%.*]] = icmp ne i64 %alloc, -1
; CHECK-NEXT:    call void @llvm.assume(i1 [[TMP]])
; CHECK-NEXT:    br i1 false, label %if.else, label %if.then
; CHECK:    call void @fortified_chk(i8* %obj, i64 %alloc)

define void @bdos_cmpm1_expr(i64 %alloc, i64 %part) {
entry:
  %sz = udiv i64 %alloc, %part
  %obj = call i8* @malloc(i64 %sz)
  %objsize = call i64 @llvm.objectsize.i64.p0i8(i8* %obj, i1 0, i1 0, i1 1)
  %cmp.not = icmp eq i64 %objsize, -1
  br i1 %cmp.not, label %if.else, label %if.then

if.then:
  call void @fortified_chk(i8* %obj, i64 %objsize)
  br label %if.end

if.else:
  call void @unfortified(i8* %obj, i64 %objsize)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK:  define void @bdos_cmpm1_expr(
; CHECK:    [[TMP:%.*]] = icmp ne i64 [[SZ:%.*]], -1
; CHECK-NEXT:    call void @llvm.assume(i1 [[TMP]])
; CHECK-NEXT:    br i1 false, label %if.else, label %if.then
; CHECK:    call void @fortified_chk(i8* %obj, i64 [[SZ]])

declare void @bury(i32) local_unnamed_addr #2

; Function Attrs: nounwind allocsize(0)
declare i8* @malloc(i64)

declare i8* @get_unknown_buffer()

; Function Attrs: nounwind
declare void @free(i8* nocapture)

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1, i1)

declare void @fortified_chk(i8*, i64)

declare void @unfortified(i8*, i64)

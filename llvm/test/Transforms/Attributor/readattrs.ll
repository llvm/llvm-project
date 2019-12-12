; RUN: opt < %s -attributor -attributor-disable=false -S -attributor-annotate-decl-cs | FileCheck %s --check-prefixes=ATTRIBUTOR
; RUN: opt < %s -aa-pipeline=basic-aa -passes='attributor' -attributor-disable=false -S -attributor-annotate-decl-cs | FileCheck %s --check-prefixes=ATTRIBUTOR
; Copied from Transforms/FunctionAttrs/readattrs.ll

@x = global i32 0

declare void @test1_1(i8* %x1_1, i8* readonly %y1_1, ...)

; NOTE: readonly for %y1_2 would be OK here but not for the similar situation in test13.
;
; ATTRIBUTOR: define void @test1_2(i8* %x1_2, i8* %y1_2, i8* %z1_2)
define void @test1_2(i8* %x1_2, i8* %y1_2, i8* %z1_2) {
  call void (i8*, i8*, ...) @test1_1(i8* %x1_2, i8* %y1_2, i8* %z1_2)
  store i32 0, i32* @x
  ret void
}

; ATTRIBUTOR: define i8* @test2(i8* nofree readnone returned %p)
define i8* @test2(i8* %p) {
  store i32 0, i32* @x
  ret i8* %p
}

; ATTRIBUTOR: define i1 @test3(i8* nofree readnone %p, i8* nofree readnone %q)
define i1 @test3(i8* %p, i8* %q) {
  %A = icmp ult i8* %p, %q
  ret i1 %A
}

declare void @test4_1(i8* nocapture) readonly

; ATTRIBUTOR: define void @test4_2(i8* nocapture readonly %p)
define void @test4_2(i8* %p) {
  call void @test4_1(i8* %p)
  ret void
}

; ATTRIBUTOR: define void @test5(i8** nocapture nofree nonnull writeonly dereferenceable(8) %p, i8* nofree writeonly %q)
; Missed optz'n: we could make %q readnone, but don't break test6!
define void @test5(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  ret void
}

declare void @test6_1()
; ATTRIBUTOR: define void @test6_2(i8** nocapture nonnull writeonly dereferenceable(8) %p, i8* %q)
; This is not a missed optz'n.
define void @test6_2(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  call void @test6_1()
  ret void
}

; ATTRIBUTOR: define void @test7_1(i32* inalloca nocapture nofree writeonly %a)
; inalloca parameters are always considered written
define void @test7_1(i32* inalloca %a) {
  ret void
}

; ATTRIBUTOR: define i32* @test8_1(i32* nofree readnone returned %p)
define i32* @test8_1(i32* %p) {
entry:
  ret i32* %p
}

; ATTRIBUTOR: define void @test8_2(i32* nocapture nofree writeonly %p)
define void @test8_2(i32* %p) {
entry:
  %call = call i32* @test8_1(i32* %p)
  store i32 10, i32* %call, align 4
  ret void
}

; ATTRIBUTOR: declare void @llvm.masked.scatter
declare void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32>%val, <4 x i32*>, i32, <4 x i1>)

; ATTRIBUTOR-NOT: readnone
; ATTRIBUTOR-NOT: readonly
; ATTRIBUTOR: define void @test9
define void @test9(<4 x i32*> %ptrs, <4 x i32>%val) {
  call void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32>%val, <4 x i32*> %ptrs, i32 4, <4 x i1><i1 true, i1 false, i1 true, i1 false>)
  ret void
}

; ATTRIBUTOR: declare <4 x i32> @llvm.masked.gather
declare <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)
; ATTRIBUTOR: readonly
; ATTRIBUTOR: define <4 x i32> @test10
define <4 x i32> @test10(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*> %ptrs, i32 4, <4 x i1><i1 true, i1 false, i1 true, i1 false>, <4 x i32>undef)
  ret <4 x i32> %res
}

; ATTRIBUTOR: declare <4 x i32> @test11_1
declare <4 x i32> @test11_1(<4 x i32*>) argmemonly nounwind readonly
; ATTRIBUTOR: readonly
; ATTRIBUTOR-NOT: readnone
; ATTRIBUTOR: define <4 x i32> @test11_2
define <4 x i32> @test11_2(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @test11_1(<4 x i32*> %ptrs)
  ret <4 x i32> %res
}

declare <4 x i32> @test12_1(<4 x i32*>) argmemonly nounwind
; ATTRIBUTOR-NOT: readnone
; ATTRIBUTOR: define <4 x i32> @test12_2
define <4 x i32> @test12_2(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @test12_1(<4 x i32*> %ptrs)
  ret <4 x i32> %res
}

; ATTRIBUTOR: define i32 @volatile_load(
; ATTRIBUTOR-NOT: readonly
; ATTRIBUTOR: ret
define i32 @volatile_load(i32* %p) {
  %load = load volatile i32, i32* %p
  ret i32 %load
}

declare void @escape_readnone_ptr(i8** %addr, i8* readnone %ptr)
declare void @escape_readonly_ptr(i8** %addr, i8* readonly %ptr)

; The argument pointer %escaped_then_written cannot be marked readnone/only even
; though the only direct use, in @escape_readnone_ptr/@escape_readonly_ptr,
; is marked as readnone/only. However, the functions can write the pointer into
; %addr, causing the store to write to %escaped_then_written.
;
;
; ATTRIBUTOR: define void @unsound_readnone(i8* nocapture nofree readnone %ignored, i8* %escaped_then_written)
; ATTRIBUTOR: define void @unsound_readonly(i8* nocapture nofree readnone %ignored, i8* %escaped_then_written)
define void @unsound_readnone(i8* %ignored, i8* %escaped_then_written) {
  %addr = alloca i8*
  call void @escape_readnone_ptr(i8** %addr, i8* %escaped_then_written)
  %addr.ld = load i8*, i8** %addr
  store i8 0, i8* %addr.ld
  ret void
}

define void @unsound_readonly(i8* %ignored, i8* %escaped_then_written) {
  %addr = alloca i8*
  call void @escape_readonly_ptr(i8** %addr, i8* %escaped_then_written)
  %addr.ld = load i8*, i8** %addr
  store i8 0, i8* %addr.ld
  ret void
}

; Byval but not readonly/none tests
;
;{
declare void @escape_i8(i8* %ptr)

; ATTRIBUTOR:      @byval_not_readonly_1
; ATTRIBUTOR-SAME: i8* byval %written
define void @byval_not_readonly_1(i8* byval %written) readonly {
  call void @escape_i8(i8* %written)
  ret void
}

; ATTRIBUTOR:      @byval_not_readonly_2
; ATTRIBUTOR-SAME: i8* nocapture nofree nonnull writeonly byval dereferenceable(1) %written
define void @byval_not_readonly_2(i8* byval %written) readonly {
  store i8 0, i8* %written
  ret void
}

; ATTRIBUTOR:      @byval_not_readnone_1
; ATTRIBUTOR-SAME: i8* byval %written
define void @byval_not_readnone_1(i8* byval %written) readnone {
  call void @escape_i8(i8* %written)
  ret void
}

; ATTRIBUTOR:      @byval_not_readnone_2
; ATTRIBUTOR-SAME: i8* nocapture nofree nonnull writeonly byval dereferenceable(1) %written
define void @byval_not_readnone_2(i8* byval %written) readnone {
  store i8 0, i8* %written
  ret void
}

; ATTRIBUTOR:      @byval_no_fnarg
; ATTRIBUTOR-SAME: i8* nocapture nofree nonnull writeonly byval dereferenceable(1) %written
define void @byval_no_fnarg(i8* byval %written) {
  store i8 0, i8* %written
  ret void
}

; ATTRIBUTOR: @testbyval
; ATTRIBUTOR-SAME: i8* nocapture readonly %read_only
define void @testbyval(i8* %read_only) {
  call void @byval_not_readonly_1(i8* %read_only)
  call void @byval_not_readonly_2(i8* %read_only)
  call void @byval_not_readnone_1(i8* %read_only)
  call void @byval_not_readnone_2(i8* %read_only)
  call void @byval_no_fnarg(i8* %read_only)
  ret void
}
;}

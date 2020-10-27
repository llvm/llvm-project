; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes='cgscc(function-attrs)' -S | FileCheck %s

@x = global i32 0

declare void @test1_1(i8* %x1_1, i8* readonly %y1_1, ...)

; NOTE: readonly for %y1_2 would be OK here but not for the similar situation in test13.
;
; CHECK: define void @test1_2(i8* %x1_2, i8* readonly %y1_2, i8* %z1_2)
define void @test1_2(i8* %x1_2, i8* %y1_2, i8* %z1_2) {
  call void (i8*, i8*, ...) @test1_1(i8* %x1_2, i8* %y1_2, i8* %z1_2)
  store i32 0, i32* @x
  ret void
}

; CHECK: define i8* @test2(i8* readnone returned %p)
define i8* @test2(i8* %p) {
  store i32 0, i32* @x
  ret i8* %p
}

; CHECK: define i1 @test3(i8* readnone %p, i8* readnone %q)
define i1 @test3(i8* %p, i8* %q) {
  %A = icmp ult i8* %p, %q
  ret i1 %A
}

declare void @test4_1(i8* nocapture) readonly

; CHECK: define void @test4_2(i8* nocapture readonly %p)
define void @test4_2(i8* %p) {
  call void @test4_1(i8* %p)
  ret void
}

; CHECK: define void @test5(i8** nocapture %p, i8* %q)
; Missed optz'n: we could make %q readnone, but don't break test6!
define void @test5(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  ret void
}

declare void @test6_1()
; CHECK: define void @test6_2(i8** nocapture %p, i8* %q)
; This is not a missed optz'n.
define void @test6_2(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  call void @test6_1()
  ret void
}

; CHECK: define void @test7_1(i32* inalloca nocapture %a)
; inalloca parameters are always considered written
define void @test7_1(i32* inalloca %a) {
  ret void
}

; CHECK: define void @test7_2(i32* nocapture preallocated(i32) %a)
; preallocated parameters are always considered written
define void @test7_2(i32* preallocated(i32) %a) {
  ret void
}

; CHECK: define i32* @test8_1(i32* readnone returned %p)
define i32* @test8_1(i32* %p) {
entry:
  ret i32* %p
}

; CHECK: define void @test8_2(i32* %p)
define void @test8_2(i32* %p) {
entry:
  %call = call i32* @test8_1(i32* %p)
  store i32 10, i32* %call, align 4
  ret void
}

; CHECK: declare void @llvm.masked.scatter
declare void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32>%val, <4 x i32*>, i32, <4 x i1>)

; CHECK-NOT: readnone
; CHECK-NOT: readonly
; CHECK: define void @test9
define void @test9(<4 x i32*> %ptrs, <4 x i32>%val) {
  call void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32>%val, <4 x i32*> %ptrs, i32 4, <4 x i1><i1 true, i1 false, i1 true, i1 false>)
  ret void
}

; CHECK: declare <4 x i32> @llvm.masked.gather
declare <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)
; CHECK: readonly
; CHECK: define <4 x i32> @test10
define <4 x i32> @test10(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*> %ptrs, i32 4, <4 x i1><i1 true, i1 false, i1 true, i1 false>, <4 x i32>undef)
  ret <4 x i32> %res
}

; CHECK: declare <4 x i32> @test11_1
declare <4 x i32> @test11_1(<4 x i32*>) argmemonly nounwind readonly
; CHECK: readonly
; CHECK-NOT: readnone
; CHECK: define <4 x i32> @test11_2
define <4 x i32> @test11_2(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @test11_1(<4 x i32*> %ptrs)
  ret <4 x i32> %res
}

declare <4 x i32> @test12_1(<4 x i32*>) argmemonly nounwind
; CHECK-NOT: readnone
; CHECK: define <4 x i32> @test12_2
define <4 x i32> @test12_2(<4 x i32*> %ptrs) {
  %res = call <4 x i32> @test12_1(<4 x i32*> %ptrs)
  ret <4 x i32> %res
}

; CHECK: define i32 @volatile_load(
; CHECK-NOT: readonly
; CHECK: ret
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
; FIXME: This test currently exposes a bug in function-attrs!
;
; CHECK: define void @unsound_readnone(i8* nocapture readnone %ignored, i8* readnone %escaped_then_written)
; CHECK: define void @unsound_readonly(i8* nocapture readnone %ignored, i8* readonly %escaped_then_written)
;
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

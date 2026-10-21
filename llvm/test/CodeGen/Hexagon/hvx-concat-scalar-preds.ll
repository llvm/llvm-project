; RUN: llc -mtriple=hexagon -mattr=+hvxv75,+hvx-length128b < %s | FileCheck %s
; REQUIRES: asserts


; CHECK-LABEL: test:
; CHECK: dealloc_return
define void @test(ptr %aptr, ptr %cptr, i32 %T, i32 %W) #0 {
  %T.splatinsert = insertelement <8 x i32> poison, i32 %T, i64 0
  %T.splat = shufflevector <8 x i32> %T.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %cmp19.ls = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %T.splat
  %cmp19.bcast = shufflevector <8 x i1> %cmp19.ls, <8 x i1> poison,
      <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
                  i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2,
                  i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3,
                  i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4,
                  i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5,
                  i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6,
                  i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %bc0 = bitcast <8 x i1> %cmp19.ls to i8
  %cmp19.red = icmp ne i8 %bc0, 0
  %cmp19.red.splatinsert = insertelement <8 x i1> poison, i1 %cmp19.red, i64 0
  %cmp19.red.splat = shufflevector <8 x i1> %cmp19.red.splatinsert, <8 x i1> poison, <8 x i32> zeroinitializer
  %W.splatinsert = insertelement <8 x i32> poison, i32 %W, i64 0
  %W.splat = shufflevector <8 x i32> %W.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %cmp43.ls = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %W.splat
  %cmp43.bcast = shufflevector <8 x i1> %cmp43.ls, <8 x i1> poison,
      <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %sel = select <8 x i1> %cmp19.red.splat, <8 x i1> %cmp43.ls, <8 x i1> zeroinitializer
  %bc1 = bitcast <8 x i1> %sel to i8
  %cmp43.red = icmp ne i8 %bc1, 0
  %cmp43.red.splatinsert = insertelement <8 x i1> poison, i1 %cmp43.red, i64 0
  %cmp43.red.splat = shufflevector <8 x i1> %cmp43.red.splatinsert, <8 x i1> poison, <8 x i32> zeroinitializer
  %mask = and <64 x i1> %cmp43.bcast, %cmp19.bcast
  %gep = getelementptr i8, ptr %cptr,
      <64 x i32> <i32 0,  i32 1,  i32 2,  i32 3,  i32 4,  i32 5,  i32 6,  i32 7,
                  i32 1,  i32 2,  i32 3,  i32 4,  i32 5,  i32 6,  i32 7,  i32 8,
                  i32 2,  i32 3,  i32 4,  i32 5,  i32 6,  i32 7,  i32 8,  i32 9,
                  i32 3,  i32 4,  i32 5,  i32 6,  i32 7,  i32 8,  i32 9,  i32 10,
                  i32 4,  i32 5,  i32 6,  i32 7,  i32 8,  i32 9,  i32 10, i32 11,
                  i32 5,  i32 6,  i32 7,  i32 8,  i32 9,  i32 10, i32 11, i32 12,
                  i32 6,  i32 7,  i32 8,  i32 9,  i32 10, i32 11, i32 12, i32 13,
                  i32 7,  i32 8,  i32 9,  i32 10, i32 11, i32 12, i32 13, i32 14>
  %data = load <64 x i8>, ptr %aptr, align 1
  tail call void @llvm.masked.scatter.v64i8.v64p0(<64 x i8> %data, <64 x ptr> %gep, <64 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v64i8.v64p0(<64 x i8>, <64 x ptr>, <64 x i1>)

attributes #0 = { "target-cpu"="hexagonv75" "target-features"="+hvx-length128b,+hvxv75,+v75" }

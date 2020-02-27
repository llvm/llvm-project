; RUN: llc -march=hexagon  < %s | FileCheck %s

; Test that LLVM does not assert and bitcast v64i1 to i64 is lowered.

; CHECK: v[[REG1:[0-9]+]] = valign(v{{[0-9]+}},v{{[0-9]+}},#2)
; CHECK: v[[REG2:[0-9]+]] = vor(v{{[0-9]+}},v[[REG1]])
; CHECK: v[[REG3:[0-9]+]] = valign(v[[REG2]],v[[REG2]],#1)
; CHECK: v[[REG4:[0-9]+]] = vor(v{{[0-9]+}},v[[REG3]])
; CHECK: v[[REG5:[0-9]+]] = vand(v[[REG4]],v{{[0-9]+}})
; CHECK: v{{[0-9]+}}.w = vasl(v[[REG5]].w,v{{[0-9]+}}.w)

target triple = "hexagon"

define dso_local void @fun() local_unnamed_addr #0 {
entry:
  br i1 undef, label %cleanup, label %if.end

if.end:
  %0 = load i8, i8* undef, align 1
  %conv13.i = zext i8 %0 to i32
  %trip.count.minus.1216 = add nsw i32 %conv13.i, -1
  %broadcast.splatinsert221 = insertelement <64 x i32> undef, i32 %trip.count.minus.1216, i32 0
  %broadcast.splat222 = shufflevector <64 x i32> %broadcast.splatinsert221, <64 x i32> undef, <64 x i32> zeroinitializer
  %1 = icmp ule <64 x i32> undef, %broadcast.splat222
  %wide.masked.load223 = call <64 x i8> @llvm.masked.load.v64i8.p0v64i8(<64 x i8>* nonnull undef, i32 1, <64 x i1> %1, <64 x i8> undef)
  %2 = lshr <64 x i8> %wide.masked.load223, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %3 = and <64 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %4 = zext <64 x i8> %3 to <64 x i32>
  %5 = add nsw <64 x i32> undef, %4
  %6 = select <64 x i1> %1, <64 x i32> %5, <64 x i32> undef
  %bin.rdx225 = add <64 x i32> %6, undef
  %bin.rdx227 = add <64 x i32> %bin.rdx225, undef
  %bin.rdx229 = add <64 x i32> %bin.rdx227, undef
  %bin.rdx231 = add <64 x i32> %bin.rdx229, undef
  %bin.rdx233 = add <64 x i32> %bin.rdx231, undef
  %bin.rdx235 = add <64 x i32> %bin.rdx233, undef
  %bin.rdx237 = add <64 x i32> %bin.rdx235, undef
  %7 = extractelement <64 x i32> %bin.rdx237, i32 0
  %nChans = getelementptr inbounds i8, i8* null, i32 2160
  %8 = bitcast i8* %nChans to i32*
  store i32 %7, i32* %8, align 4
  br label %cleanup

cleanup:
  ret void
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <64 x i8> @llvm.masked.load.v64i8.p0v64i8(<64 x i8>*, i32, <64 x i1>, <64 x i8>)

attributes #0 = { "target-features"="+hvx-length64b,+hvxv67,+v67,-long-calls" }

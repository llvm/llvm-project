; RUN: llc  < %s | FileCheck %s

; This test checks that store a vector predicate of type v128i1 is lowered
; and two double stores are generated.

; CHECK-DAG: memd(r{{[0-9]+}}+#0) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK-DAG: memd(r{{[0-9]+}}+#8) = r{{[0-9]+}}:{{[0-9]+}}

target triple = "hexagon"

define dso_local void @raac_UnpackADIFHeader() local_unnamed_addr #0 {
entry:
  br i1 undef, label %cleanup, label %if.end

if.end:
  %0 = load i8, i8* undef, align 1
  %conv13.i = zext i8 %0 to i32
  %trip.count.minus.1216 = add nsw i32 %conv13.i, -1
  %broadcast.splatinsert221 = insertelement <128 x i32> undef, i32 %trip.count.minus.1216, i32 0
  %broadcast.splat222 = shufflevector <128 x i32> %broadcast.splatinsert221, <128 x i32> undef, <128 x i32> zeroinitializer
  %1 = icmp ule <128 x i32> undef, %broadcast.splat222
  %wide.masked.load223 = call <128 x i8> @llvm.masked.load.v128i8.p0v128i8(<128 x i8>* nonnull undef, i32 1, <128 x i1> %1, <128 x i8> undef)
  %2 = lshr <128 x i8> %wide.masked.load223, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %3 = and <128 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %4 = zext <128 x i8> %3 to <128 x i32>
  %5 = add nsw <128 x i32> undef, %4
  %6 = select <128 x i1> %1, <128 x i32> %5, <128 x i32> undef
  %bin.rdx225 = add <128 x i32> %6, undef
  %bin.rdx227 = add <128 x i32> %bin.rdx225, undef
  %bin.rdx229 = add <128 x i32> %bin.rdx227, undef
  %bin.rdx231 = add <128 x i32> %bin.rdx229, undef
  %bin.rdx233 = add <128 x i32> %bin.rdx231, undef
  %bin.rdx235 = add <128 x i32> %bin.rdx233, undef
  %bin.rdx237 = add <128 x i32> %bin.rdx235, undef
  %7 = extractelement <128 x i32> %bin.rdx237, i32 0
  %nChans = getelementptr inbounds i8, i8* null, i32 2160
  %8 = bitcast i8* %nChans to i32*
  store i32 %7, i32* %8, align 4
  br label %cleanup

cleanup:
  ret void
  }

declare <128 x i8> @llvm.masked.load.v128i8.p0v128i8(<128 x i8>*, i32 immarg, <128 x i1>, <128 x i8>)

attributes #0 = { "target-features"="+hvx-length128b,+hvxv67,+v67,-long-calls" }

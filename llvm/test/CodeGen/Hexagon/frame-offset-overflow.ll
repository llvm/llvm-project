; REQUIRES: asserts
; RUN: llc -march=hexagon --stats -o - 2>&1 < %s | FileCheck %s

; Check that the compilation succeeded and that some code was generated.
; CHECK: vadd

; Check that the loop is pipelined and that a valid node order is used.
; CHECK-NOT: Number of node order issues found
; CHECK: Number of loops software pipelined
; CHECK-NOT: Number of node order issues found

target triple = "hexagon"

define void @fred(i16* noalias nocapture readonly %p0, i32 %p1, i32 %p2, i16* noalias nocapture %p3, i32 %p4) local_unnamed_addr #1 {
entry:
  %mul = mul i32 %p4, %p1
  %add.ptr = getelementptr inbounds i16, i16* %p0, i32 %mul
  %add = add nsw i32 %p4, 1
  %rem = srem i32 %add, 5
  %mul1 = mul i32 %rem, %p1
  %add.ptr2 = getelementptr inbounds i16, i16* %p0, i32 %mul1
  %add.ptr6 = getelementptr inbounds i16, i16* %p0, i32 0
  %add7 = add nsw i32 %p4, 3
  %rem8 = srem i32 %add7, 5
  %mul9 = mul i32 %rem8, %p1
  %add.ptr10 = getelementptr inbounds i16, i16* %p0, i32 %mul9
  %add.ptr14 = getelementptr inbounds i16, i16* %p0, i32 0
  %incdec.ptr18 = getelementptr inbounds i16, i16* %add.ptr14, i32 32
  %0 = bitcast i16* %incdec.ptr18 to <16 x i32>*
  %incdec.ptr17 = getelementptr inbounds i16, i16* %add.ptr10, i32 32
  %1 = bitcast i16* %incdec.ptr17 to <16 x i32>*
  %incdec.ptr16 = getelementptr inbounds i16, i16* %add.ptr6, i32 32
  %2 = bitcast i16* %incdec.ptr16 to <16 x i32>*
  %incdec.ptr15 = getelementptr inbounds i16, i16* %add.ptr2, i32 32
  %3 = bitcast i16* %incdec.ptr15 to <16 x i32>*
  %incdec.ptr = getelementptr inbounds i16, i16* %add.ptr, i32 32
  %4 = bitcast i16* %incdec.ptr to <16 x i32>*
  %5 = bitcast i16* %p3 to <16 x i32>*
  br i1 undef, label %for.end.loopexit.unr-lcssa, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %optr.0102 = phi <16 x i32>* [ %incdec.ptr24.3, %for.body ], [ %5, %entry ]
  %iptr4.0101 = phi <16 x i32>* [ %incdec.ptr23.3, %for.body ], [ %0, %entry ]
  %iptr3.0100 = phi <16 x i32>* [ %incdec.ptr22.3, %for.body ], [ %1, %entry ]
  %iptr2.099 = phi <16 x i32>* [ undef, %for.body ], [ %2, %entry ]
  %iptr1.098 = phi <16 x i32>* [ %incdec.ptr20.3, %for.body ], [ %3, %entry ]
  %iptr0.097 = phi <16 x i32>* [ %incdec.ptr19.3, %for.body ], [ %4, %entry ]
  %dVsumv1.096 = phi <32 x i32> [ %66, %for.body ], [ undef, %entry ]
  %niter = phi i32 [ %niter.nsub.3, %for.body ], [ undef, %entry ]
  %6 = load <16 x i32>, <16 x i32>* %iptr0.097, align 64, !tbaa !1
  %7 = load <16 x i32>, <16 x i32>* %iptr1.098, align 64, !tbaa !1
  %8 = load <16 x i32>, <16 x i32>* %iptr2.099, align 64, !tbaa !1
  %9 = load <16 x i32>, <16 x i32>* %iptr3.0100, align 64, !tbaa !1
  %10 = load <16 x i32>, <16 x i32>* %iptr4.0101, align 64, !tbaa !1
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %6, <16 x i32> %10)
  %12 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %11, <16 x i32> %8, i32 393222)
  %13 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %9, <16 x i32> %7)
  %14 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %12, <32 x i32> %13, i32 67372036)
  %15 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %dVsumv1.096)
  %16 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %14)
  %17 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %16, <16 x i32> %15, i32 4)
  %18 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %14)
  %19 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %16, <16 x i32> %15, i32 8)
  %20 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %18, <16 x i32> undef, i32 8)
  %21 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %17, <16 x i32> %19)
  %22 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %15, <16 x i32> %19)
  %23 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %22, <16 x i32> %17, i32 101058054)
  %24 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %23, <16 x i32> zeroinitializer, i32 67372036)
  %25 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> %20)
  %26 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %25, <16 x i32> undef, i32 101058054)
  %27 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %26, <16 x i32> %21, i32 67372036)
  %28 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %27, <16 x i32> %24, i32 8)
  %incdec.ptr24 = getelementptr inbounds <16 x i32>, <16 x i32>* %optr.0102, i32 1
  store <16 x i32> %28, <16 x i32>* %optr.0102, align 64, !tbaa !1
  %incdec.ptr19.1 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr0.097, i32 2
  %incdec.ptr23.1 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr4.0101, i32 2
  %29 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %14)
  %30 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %14)
  %31 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> %29, i32 4)
  %32 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> %30, i32 4)
  %33 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %29, i32 8)
  %34 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %30, i32 8)
  %35 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %31, <16 x i32> %33)
  %36 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %29, <16 x i32> %33)
  %37 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %36, <16 x i32> %31, i32 101058054)
  %38 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %37, <16 x i32> undef, i32 67372036)
  %39 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %30, <16 x i32> %34)
  %40 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %39, <16 x i32> %32, i32 101058054)
  %41 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %40, <16 x i32> %35, i32 67372036)
  %42 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %41, <16 x i32> %38, i32 8)
  %incdec.ptr24.1 = getelementptr inbounds <16 x i32>, <16 x i32>* %optr.0102, i32 2
  store <16 x i32> %42, <16 x i32>* %incdec.ptr24, align 64, !tbaa !1
  %incdec.ptr19.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr0.097, i32 3
  %43 = load <16 x i32>, <16 x i32>* %incdec.ptr19.1, align 64, !tbaa !1
  %incdec.ptr20.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr1.098, i32 3
  %incdec.ptr21.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr2.099, i32 3
  %incdec.ptr22.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr3.0100, i32 3
  %incdec.ptr23.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr4.0101, i32 3
  %44 = load <16 x i32>, <16 x i32>* %incdec.ptr23.1, align 64, !tbaa !1
  %45 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %43, <16 x i32> %44)
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %45, <16 x i32> undef, i32 393222)
  %47 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %46, <32 x i32> undef, i32 67372036)
  %48 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %47)
  %49 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %48, <16 x i32> undef, i32 4)
  %50 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %48, <16 x i32> undef, i32 8)
  %51 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> zeroinitializer, <16 x i32> undef)
  %52 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %49, <16 x i32> %50)
  %53 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> %50)
  %54 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %53, <16 x i32> %49, i32 101058054)
  %55 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %54, <16 x i32> %51, i32 67372036)
  %56 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> undef, <16 x i32> %52, i32 67372036)
  %57 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %56, <16 x i32> %55, i32 8)
  %incdec.ptr24.2 = getelementptr inbounds <16 x i32>, <16 x i32>* %optr.0102, i32 3
  store <16 x i32> %57, <16 x i32>* %incdec.ptr24.1, align 64, !tbaa !1
  %incdec.ptr19.3 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr0.097, i32 4
  %58 = load <16 x i32>, <16 x i32>* %incdec.ptr19.2, align 64, !tbaa !1
  %incdec.ptr20.3 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr1.098, i32 4
  %59 = load <16 x i32>, <16 x i32>* %incdec.ptr20.2, align 64, !tbaa !1
  %60 = load <16 x i32>, <16 x i32>* %incdec.ptr21.2, align 64, !tbaa !1
  %incdec.ptr22.3 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr3.0100, i32 4
  %61 = load <16 x i32>, <16 x i32>* %incdec.ptr22.2, align 64, !tbaa !1
  %incdec.ptr23.3 = getelementptr inbounds <16 x i32>, <16 x i32>* %iptr4.0101, i32 4
  %62 = load <16 x i32>, <16 x i32>* %incdec.ptr23.2, align 64, !tbaa !1
  %63 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %58, <16 x i32> %62)
  %64 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %63, <16 x i32> %60, i32 393222)
  %65 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %61, <16 x i32> %59)
  %66 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %64, <32 x i32> %65, i32 67372036)
  %67 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %47)
  %68 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %66)
  %69 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %68, <16 x i32> undef, i32 4)
  %70 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %66)
  %71 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %70, <16 x i32> %67, i32 4)
  %72 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %70, <16 x i32> %67, i32 8)
  %73 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %67, <16 x i32> %71)
  %74 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> undef, <16 x i32> %69, i32 101058054)
  %75 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %74, <16 x i32> %73, i32 67372036)
  %76 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %67, <16 x i32> %72)
  %77 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %76, <16 x i32> %71, i32 101058054)
  %78 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %77, <16 x i32> undef, i32 67372036)
  %79 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %78, <16 x i32> %75, i32 8)
  %incdec.ptr24.3 = getelementptr inbounds <16 x i32>, <16 x i32>* %optr.0102, i32 4
  store <16 x i32> %79, <16 x i32>* %incdec.ptr24.2, align 64, !tbaa !1
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.end.loopexit.unr-lcssa, label %for.body

for.end.loopexit.unr-lcssa:                       ; preds = %for.body, %entry
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32>, <16 x i32>, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}

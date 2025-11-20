; REQUIRES: asserts
; RUN: llc -mtriple=hexagon --stats -o - 2>&1 < %s | FileCheck %s

; Check that the compilation succeeded and that some code was generated.
; CHECK: vadd

; Check that the loop is pipelined and that a valid node order is used.
; CHECK-NOT: Number of node order issues found
; CHECK: Number of loops software pipelined
; CHECK-NOT: Number of node order issues found

target triple = "hexagon"

define void @fred(ptr noalias nocapture readonly %p0, i32 %p1, i32 %p2, ptr noalias nocapture %p3, i32 %p4) local_unnamed_addr #1 {
entry:
  %mul = mul i32 %p4, %p1
  %add.ptr = getelementptr inbounds i16, ptr %p0, i32 %mul
  %add = add nsw i32 %p4, 1
  %rem = srem i32 %add, 5
  %mul1 = mul i32 %rem, %p1
  %add.ptr2 = getelementptr inbounds i16, ptr %p0, i32 %mul1
  %add7 = add nsw i32 %p4, 3
  %rem8 = srem i32 %add7, 5
  %mul9 = mul i32 %rem8, %p1
  %add.ptr10 = getelementptr inbounds i16, ptr %p0, i32 %mul9
  %incdec.ptr18 = getelementptr inbounds i16, ptr %p0, i32 32
  %incdec.ptr17 = getelementptr inbounds i16, ptr %add.ptr10, i32 32
  %incdec.ptr16 = getelementptr inbounds i16, ptr %p0, i32 32
  %incdec.ptr15 = getelementptr inbounds i16, ptr %add.ptr2, i32 32
  %incdec.ptr = getelementptr inbounds i16, ptr %add.ptr, i32 32
  br i1 undef, label %for.end.loopexit.unr-lcssa, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %optr.0102 = phi ptr [ %incdec.ptr24.3, %for.body ], [ %p3, %entry ]
  %iptr4.0101 = phi ptr [ %incdec.ptr23.3, %for.body ], [ %incdec.ptr18, %entry ]
  %iptr3.0100 = phi ptr [ %incdec.ptr22.3, %for.body ], [ %incdec.ptr17, %entry ]
  %iptr2.099 = phi ptr [ poison, %for.body ], [ %incdec.ptr16, %entry ]
  %iptr1.098 = phi ptr [ %incdec.ptr20.3, %for.body ], [ %incdec.ptr15, %entry ]
  %iptr0.097 = phi ptr [ %incdec.ptr19.3, %for.body ], [ %incdec.ptr, %entry ]
  %dVsumv1.096 = phi <32 x i32> [ %60, %for.body ], [ undef, %entry ]
  %niter = phi i32 [ %niter.nsub.3, %for.body ], [ undef, %entry ]
  %0 = load <16 x i32>, ptr %iptr0.097, align 64, !tbaa !1
  %1 = load <16 x i32>, ptr %iptr1.098, align 64, !tbaa !1
  %2 = load <16 x i32>, ptr %iptr2.099, align 64, !tbaa !1
  %3 = load <16 x i32>, ptr %iptr3.0100, align 64, !tbaa !1
  %4 = load <16 x i32>, ptr %iptr4.0101, align 64, !tbaa !1
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %0, <16 x i32> %4)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %5, <16 x i32> %2, i32 393222)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %3, <16 x i32> %1)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %6, <32 x i32> %7, i32 67372036)
  %9 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %dVsumv1.096)
  %10 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %8)
  %11 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %10, <16 x i32> %9, i32 4)
  %12 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %8)
  %13 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %10, <16 x i32> %9, i32 8)
  %14 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %12, <16 x i32> undef, i32 8)
  %15 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %11, <16 x i32> %13)
  %16 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %9, <16 x i32> %13)
  %17 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %16, <16 x i32> %11, i32 101058054)
  %18 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %17, <16 x i32> zeroinitializer, i32 67372036)
  %19 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> %14)
  %20 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %19, <16 x i32> undef, i32 101058054)
  %21 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %20, <16 x i32> %15, i32 67372036)
  %22 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %21, <16 x i32> %18, i32 8)
  %incdec.ptr24 = getelementptr inbounds <16 x i32>, ptr %optr.0102, i32 1
  store <16 x i32> %22, ptr %optr.0102, align 64, !tbaa !1
  %incdec.ptr19.1 = getelementptr inbounds <16 x i32>, ptr %iptr0.097, i32 2
  %incdec.ptr23.1 = getelementptr inbounds <16 x i32>, ptr %iptr4.0101, i32 2
  %23 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %8)
  %24 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %8)
  %25 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> %23, i32 4)
  %26 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> %24, i32 4)
  %27 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %23, i32 8)
  %28 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %24, i32 8)
  %29 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %25, <16 x i32> %27)
  %30 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %23, <16 x i32> %27)
  %31 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %30, <16 x i32> %25, i32 101058054)
  %32 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %31, <16 x i32> undef, i32 67372036)
  %33 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %24, <16 x i32> %28)
  %34 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %33, <16 x i32> %26, i32 101058054)
  %35 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %34, <16 x i32> %29, i32 67372036)
  %36 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %35, <16 x i32> %32, i32 8)
  %incdec.ptr24.1 = getelementptr inbounds <16 x i32>, ptr %optr.0102, i32 2
  store <16 x i32> %36, ptr %incdec.ptr24, align 64, !tbaa !1
  %incdec.ptr19.2 = getelementptr inbounds <16 x i32>, ptr %iptr0.097, i32 3
  %37 = load <16 x i32>, ptr %incdec.ptr19.1, align 64, !tbaa !1
  %incdec.ptr20.2 = getelementptr inbounds <16 x i32>, ptr %iptr1.098, i32 3
  %incdec.ptr21.2 = getelementptr inbounds <16 x i32>, ptr %iptr2.099, i32 3
  %incdec.ptr22.2 = getelementptr inbounds <16 x i32>, ptr %iptr3.0100, i32 3
  %incdec.ptr23.2 = getelementptr inbounds <16 x i32>, ptr %iptr4.0101, i32 3
  %38 = load <16 x i32>, ptr %incdec.ptr23.1, align 64, !tbaa !1
  %39 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %37, <16 x i32> %38)
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %39, <16 x i32> undef, i32 393222)
  %41 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %40, <32 x i32> undef, i32 67372036)
  %42 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %41)
  %43 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %42, <16 x i32> undef, i32 4)
  %44 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %42, <16 x i32> undef, i32 8)
  %45 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> zeroinitializer, <16 x i32> undef)
  %46 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %43, <16 x i32> %44)
  %47 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> %44)
  %48 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %47, <16 x i32> %43, i32 101058054)
  %49 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %48, <16 x i32> %45, i32 67372036)
  %50 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> undef, <16 x i32> %46, i32 67372036)
  %51 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %50, <16 x i32> %49, i32 8)
  %incdec.ptr24.2 = getelementptr inbounds <16 x i32>, ptr %optr.0102, i32 3
  store <16 x i32> %51, ptr %incdec.ptr24.1, align 64, !tbaa !1
  %incdec.ptr19.3 = getelementptr inbounds <16 x i32>, ptr %iptr0.097, i32 4
  %52 = load <16 x i32>, ptr %incdec.ptr19.2, align 64, !tbaa !1
  %incdec.ptr20.3 = getelementptr inbounds <16 x i32>, ptr %iptr1.098, i32 4
  %53 = load <16 x i32>, ptr %incdec.ptr20.2, align 64, !tbaa !1
  %54 = load <16 x i32>, ptr %incdec.ptr21.2, align 64, !tbaa !1
  %incdec.ptr22.3 = getelementptr inbounds <16 x i32>, ptr %iptr3.0100, i32 4
  %55 = load <16 x i32>, ptr %incdec.ptr22.2, align 64, !tbaa !1
  %incdec.ptr23.3 = getelementptr inbounds <16 x i32>, ptr %iptr4.0101, i32 4
  %56 = load <16 x i32>, ptr %incdec.ptr23.2, align 64, !tbaa !1
  %57 = tail call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %52, <16 x i32> %56)
  %58 = tail call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %57, <16 x i32> %54, i32 393222)
  %59 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %55, <16 x i32> %53)
  %60 = tail call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %58, <32 x i32> %59, i32 67372036)
  %61 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %41)
  %62 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %60)
  %63 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %62, <16 x i32> undef, i32 4)
  %64 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %60)
  %65 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %64, <16 x i32> %61, i32 4)
  %66 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %64, <16 x i32> %61, i32 8)
  %67 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %61, <16 x i32> %65)
  %68 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> undef, <16 x i32> %63, i32 101058054)
  %69 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %68, <16 x i32> %67, i32 67372036)
  %70 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %61, <16 x i32> %66)
  %71 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %70, <16 x i32> %65, i32 101058054)
  %72 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %71, <16 x i32> undef, i32 67372036)
  %73 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %72, <16 x i32> %69, i32 8)
  %incdec.ptr24.3 = getelementptr inbounds <16 x i32>, ptr %optr.0102, i32 4
  store <16 x i32> %73, ptr %incdec.ptr24.2, align 64, !tbaa !1
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

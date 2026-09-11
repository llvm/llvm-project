; Checks for presence of any mismatch, ie. an use of qf operand
; as a sf/hf type or vice versa

; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv79,+hvx-length128B \
; RUN: -enable-postra-xqf-check 2>&1 < %s  -o /dev/null | FileCheck %s
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv81,+hvx-length128B \
; RUN: -enable-postra-xqf-check 2>&1 < %s  -o /dev/null | FileCheck %s

; CHECK: Checking for ABI compliance for XQF post register allocation
; CHECK-NOT: Mismatch:

define i32 @qhmath_hvx_sin_af(ptr noalias noundef %input, ptr noalias noundef %output) #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A2.min(i32 0, i32 64)
  %cmp10100 = icmp sgt i32 %0, 0
  br i1 %cmp10100, label %for.body12.lr.ph, label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.body12, %entry
  ret i32 0

for.body12.lr.ph:                                 ; preds = %entry
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1065353216)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -2147483648)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1067645315)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %1, <32 x i32> zeroinitializer)
  %5 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -1090519040)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %5, <32 x i32> zeroinitializer)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1026206373)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %7, <32 x i32> zeroinitializer)
  %9 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -1162475884)
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %9, <32 x i32> zeroinitializer)
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %12 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 2147483647)
  br label %for.body12

for.body12:                                       ; preds = %for.body12, %for.body12.lr.ph
  %j.0104 = phi i32 [ 0, %for.body12.lr.ph ], [ %inc, %for.body12 ]
  %optr.1102 = phi ptr [ %output, %for.body12.lr.ph ], [ %incdec.ptr14, %for.body12 ]
  %sline1p.1101 = phi <32 x i32> [ zeroinitializer, %for.body12.lr.ph ], [ %13, %for.body12 ]
  %13 = load <32 x i32>, ptr null, align 128
  %14 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %13, <32 x i32> %sline1p.1101, i32 0)
  %15 = tail call <128 x i1> @llvm.hexagon.V6.vgtsf.128B(<32 x i32> %14, <32 x i32> zeroinitializer)
  %16 = tail call <32 x i32> @llvm.hexagon.V6.vxor.128B(<32 x i32> %14, <32 x i32> %2)
  %17 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %15, <32 x i32> %14, <32 x i32> %16)
  %18 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> %17, <32 x i32> %3)
  %19 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %18)
  %20 = tail call <128 x i1> @llvm.hexagon.V6.vgtuw.128B(<32 x i32> %19, <32 x i32> %12)
  %21 = lshr <32 x i32> %19, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  %and.i.i = and <32 x i32> %21, <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  %sub.i.i = add nsw <32 x i32> %and.i.i, <i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127, i32 -127>
  %22 = tail call <32 x i32> @llvm.hexagon.V6.vmaxw.128B(<32 x i32> %sub.i.i, <32 x i32> zeroinitializer)
  %sub1.i.i = sub <32 x i32> <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>, %22
  %23 = tail call <32 x i32> @llvm.hexagon.V6.vmaxw.128B(<32 x i32> %sub1.i.i, <32 x i32> zeroinitializer)
  %shl.i.i = shl nuw <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %22
  %24 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %shl.i.i)
  %25 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %20, <32 x i32> zeroinitializer, <32 x i32> %24)
  %shl5.neg.i.i = shl <32 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, %23
  %and7.i.i = and <32 x i32> %shl5.neg.i.i, %19
  %26 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %and7.i.i)
  %27 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> %26)
  %28 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %27)
  %29 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %28)
  %30 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32> %17, <32 x i32> %29)
  %31 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %25, <32 x i32> zeroinitializer)
  %32 = tail call <128 x i1> @llvm.hexagon.V6.veqw.128B(<32 x i32> %31, <32 x i32> zeroinitializer)
  %33 = tail call <128 x i1> @llvm.hexagon.V6.pred.xor.128B(<128 x i1> %15, <128 x i1> %32)
  %34 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %30)
  %35 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %34, <32 x i32> zeroinitializer)
  %36 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> %35, <32 x i32> %35)
  %37 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %11, <32 x i32> %36)
  %38 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %10, <32 x i32> %37)
  %39 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %38, <32 x i32> %36)
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %8, <32 x i32> %39)
  %41 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %40, <32 x i32> %36)
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %6, <32 x i32> %41)
  %43 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %42, <32 x i32> %36)
  %44 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %4, <32 x i32> %43)
  %45 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %44)
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %45, <32 x i32> zeroinitializer)
  %47 = tail call <32 x i32> @llvm.hexagon.V6.vxor.128B(<32 x i32> %46, <32 x i32> %2)
  %48 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %33, <32 x i32> %46, <32 x i32> %47)
  %incdec.ptr14 = getelementptr inbounds <32 x i32>, ptr %optr.1102, i32 1
  store <32 x i32> %48, ptr %optr.1102, align 4
  %inc = add nuw nsw i32 %j.0104, 1
  %exitcond.not = icmp eq i32 %inc, %0
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body12
}

declare i32 @llvm.hexagon.A2.min(i32, i32) #1
declare <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.vgtsf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vxor.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.veqw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.xor.128B(<128 x i1>, <128 x i1>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.or.128B(<128 x i1>, <128 x i1>) #1
declare <128 x i1> @llvm.hexagon.V6.vgtuw.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmaxw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32>, <32 x i32>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(i32) #1
declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #2
declare <128 x i1> @llvm.hexagon.V6.veqb.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.or.n.128B(<128 x i1>, <128 x i1>) #1
declare void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #2

uselistorder ptr @llvm.hexagon.V6.lvsplatw.128B, { 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vadd.sf.128B, { 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vxor.128B, { 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vmux.128B, { 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vmpy.qf32.sf.128B, { 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vconv.sf.qf32.128B, { 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vmpy.qf32.128B, { 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vadd.qf32.128B, { 3, 2, 1, 0 }
uselistorder ptr @llvm.hexagon.V6.vmaxw.128B, { 1, 0 }

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(write) }

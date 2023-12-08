; RUN: llc -march=hexagon -hexagon-bit=0 < %s | FileCheck %s

; This spill should be eliminated.
; CHECK-NOT: vmem(r29+#6)

define void @test(ptr noalias nocapture %key, ptr noalias nocapture %data1) #0 {
entry:
  br label %for.body

for.body:
  %pkey.0542 = phi ptr [ %key, %entry ], [ null, %for.body ]
  %pdata0.0541 = phi ptr [ null, %entry ], [ %add.ptr48, %for.body ]
  %pdata1.0540 = phi ptr [ %data1, %entry ], [ %add.ptr49, %for.body ]
  %dAccum0.0539 = phi <64 x i32> [ undef, %entry ], [ %84, %for.body ]
  %0 = load <32 x i32>, ptr %pkey.0542, align 128
  %1 = load <32 x i32>, ptr %pdata0.0541, align 128
  %2 = load <32 x i32>, ptr undef, align 128
  %arrayidx4 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 2
  %3 = load <32 x i32>, ptr %arrayidx4, align 128
  %arrayidx5 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 2
  %4 = load <32 x i32>, ptr %arrayidx5, align 128
  %5 = load <32 x i32>, ptr null, align 128
  %6 = load <32 x i32>, ptr undef, align 128
  %7 = load <32 x i32>, ptr null, align 128
  %arrayidx9 = getelementptr inbounds <32 x i32>, ptr %pkey.0542, i32 3
  %arrayidx10 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 6
  %8 = load <32 x i32>, ptr %arrayidx10, align 128
  %arrayidx12 = getelementptr inbounds <32 x i32>, ptr %pkey.0542, i32 4
  %9 = load <32 x i32>, ptr %arrayidx12, align 128
  %arrayidx13 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 8
  %arrayidx14 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 8
  %10 = load <32 x i32>, ptr %arrayidx14, align 128
  %arrayidx15 = getelementptr inbounds <32 x i32>, ptr %pkey.0542, i32 5
  %11 = load <32 x i32>, ptr %arrayidx15, align 128
  %arrayidx16 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 10
  %arrayidx17 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 10
  %12 = load <32 x i32>, ptr %arrayidx17, align 128
  %arrayidx18 = getelementptr inbounds <32 x i32>, ptr %pkey.0542, i32 6
  %13 = load <32 x i32>, ptr %arrayidx18, align 128
  %arrayidx19 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 12
  %14 = load <32 x i32>, ptr %arrayidx19, align 128
  %arrayidx20 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 12
  %15 = load <32 x i32>, ptr %arrayidx20, align 128
  %arrayidx22 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 14
  %16 = load <32 x i32>, ptr %arrayidx22, align 128
  %arrayidx23 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 14
  %17 = load <32 x i32>, ptr %arrayidx23, align 128
  %18 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %0, <32 x i32> %9)
  %19 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %18, <32 x i32> %9, <32 x i32> %0)
  %20 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %18, <32 x i32> %0, <32 x i32> %9)
  %21 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %18, <32 x i32> undef, <32 x i32> %1)
  %22 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %18, <32 x i32> %10, <32 x i32> undef)
  %23 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %5, <32 x i32> %13)
  %24 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %13, <32 x i32> %5)
  %25 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %5, <32 x i32> %13)
  %26 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %14, <32 x i32> %6)
  %27 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %6, <32 x i32> %14)
  %28 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %15, <32 x i32> %7)
  %29 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %23, <32 x i32> %7, <32 x i32> %15)
  %30 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %2, <32 x i32> %11)
  %31 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> %11, <32 x i32> %2)
  %32 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> %2, <32 x i32> %11)
  %33 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> undef, <32 x i32> %3)
  %34 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> %3, <32 x i32> undef)
  %35 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> %12, <32 x i32> %4)
  %36 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %30, <32 x i32> %4, <32 x i32> %12)
  %37 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> undef)
  %38 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> undef, <32 x i32> zeroinitializer)
  %39 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %16, <32 x i32> %8)
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %8, <32 x i32> %16)
  %41 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> %17, <32 x i32> undef)
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> zeroinitializer, <32 x i32> undef, <32 x i32> %17)
  %43 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %19, <32 x i32> %24)
  %44 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %24, <32 x i32> %19)
  %45 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %19, <32 x i32> %24)
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %26, <32 x i32> %21)
  %47 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %21, <32 x i32> %26)
  %48 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %28, <32 x i32> %22)
  %49 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %43, <32 x i32> %22, <32 x i32> %28)
  %50 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %20, <32 x i32> %25)
  %51 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %50, <32 x i32> %25, <32 x i32> %20)
  %52 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %50, <32 x i32> %20, <32 x i32> %25)
  %53 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %50, <32 x i32> %27, <32 x i32> undef)
  %54 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %50, <32 x i32> undef, <32 x i32> %29)
  %55 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %31, <32 x i32> %37)
  %56 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %55, <32 x i32> %37, <32 x i32> %31)
  %57 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %55, <32 x i32> %31, <32 x i32> %37)
  %58 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %55, <32 x i32> %39, <32 x i32> %33)
  %59 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %55, <32 x i32> %41, <32 x i32> %35)
  %60 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %32, <32 x i32> %38)
  %61 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %60, <32 x i32> %40, <32 x i32> %34)
  %62 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %60, <32 x i32> %36, <32 x i32> %42)
  %63 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %44, <32 x i32> %56)
  %64 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %63, <32 x i32> %56, <32 x i32> %44)
  %65 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %63, <32 x i32> %58, <32 x i32> %46)
  %66 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %63, <32 x i32> %59, <32 x i32> %48)
  %67 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %45, <32 x i32> %57)
  %68 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %67, <32 x i32> %49, <32 x i32> zeroinitializer)
  %69 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %51, <32 x i32> zeroinitializer)
  %70 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %69, <32 x i32> %61, <32 x i32> %53)
  %71 = tail call <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %52, <32 x i32> undef)
  %72 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %71, <32 x i32> %54, <32 x i32> %62)
  %73 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %66, <32 x i32> %65)
  %74 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %68, <32 x i32> undef)
  %75 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> zeroinitializer, <32 x i32> %70)
  %76 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %72, <32 x i32> zeroinitializer)
  %77 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %dAccum0.0539, <32 x i32> %73, i32 65537)
  %78 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %77, <32 x i32> zeroinitializer, i32 65537)
  %79 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %78, <32 x i32> zeroinitializer, i32 65537)
  %80 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %79, <32 x i32> %74, i32 65537)
  %81 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %80, <32 x i32> %75, i32 65537)
  %82 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %81, <32 x i32> zeroinitializer, i32 65537)
  %83 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %82, <32 x i32> undef, i32 65537)
  %84 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %83, <32 x i32> %76, i32 65537)
  store <32 x i32> %64, ptr %pkey.0542, align 128
  store <32 x i32> %73, ptr %pdata0.0541, align 128
  store <32 x i32> zeroinitializer, ptr %arrayidx4, align 128
  store <32 x i32> zeroinitializer, ptr undef, align 128
  store <32 x i32> zeroinitializer, ptr %arrayidx20, align 128
  store <32 x i32> zeroinitializer, ptr null, align 128
  %add.ptr48 = getelementptr inbounds <32 x i32>, ptr %pdata0.0541, i32 16
  %add.ptr49 = getelementptr inbounds <32 x i32>, ptr %pdata1.0540, i32 16
  br i1 false, label %for.end, label %for.body

for.end:
  %85 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %84)
  ret void
}

declare <128 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32>, <32 x i32>) #1

declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

declare <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32>, <32 x i32>) #1

declare <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32>, <32 x i32>, i32) #1

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
attributes #1 = { nounwind readnone }

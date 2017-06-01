; RUN: llc -march=hexagon -mcpu=hexagonv60 -enable-hexagon-hvx-double \
; RUN:     -hexagon-bit=0 < %s | FileCheck %s

; This spill should be eliminated.
; CHECK-NOT: vmem(r29+#6)

define void @test(i8* noalias nocapture %key, i8* noalias nocapture %data1) #0 {
entry:
  %0 = bitcast i8* %key to <32 x i32>*
  %1 = bitcast i8* %data1 to <32 x i32>*
  br label %for.body

for.body:
  %pkey.0542 = phi <32 x i32>* [ %0, %entry ], [ null, %for.body ]
  %pdata0.0541 = phi <32 x i32>* [ null, %entry ], [ %add.ptr48, %for.body ]
  %pdata1.0540 = phi <32 x i32>* [ %1, %entry ], [ %add.ptr49, %for.body ]
  %dAccum0.0539 = phi <64 x i32> [ undef, %entry ], [ %86, %for.body ]
  %2 = load <32 x i32>, <32 x i32>* %pkey.0542, align 128
  %3 = load <32 x i32>, <32 x i32>* %pdata0.0541, align 128
  %4 = load <32 x i32>, <32 x i32>* undef, align 128
  %arrayidx4 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 2
  %5 = load <32 x i32>, <32 x i32>* %arrayidx4, align 128
  %arrayidx5 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 2
  %6 = load <32 x i32>, <32 x i32>* %arrayidx5, align 128
  %7 = load <32 x i32>, <32 x i32>* null, align 128
  %8 = load <32 x i32>, <32 x i32>* undef, align 128
  %9 = load <32 x i32>, <32 x i32>* null, align 128
  %arrayidx9 = getelementptr inbounds <32 x i32>, <32 x i32>* %pkey.0542, i32 3
  %arrayidx10 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 6
  %10 = load <32 x i32>, <32 x i32>* %arrayidx10, align 128
  %arrayidx12 = getelementptr inbounds <32 x i32>, <32 x i32>* %pkey.0542, i32 4
  %11 = load <32 x i32>, <32 x i32>* %arrayidx12, align 128
  %arrayidx13 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 8
  %arrayidx14 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 8
  %12 = load <32 x i32>, <32 x i32>* %arrayidx14, align 128
  %arrayidx15 = getelementptr inbounds <32 x i32>, <32 x i32>* %pkey.0542, i32 5
  %13 = load <32 x i32>, <32 x i32>* %arrayidx15, align 128
  %arrayidx16 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 10
  %arrayidx17 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 10
  %14 = load <32 x i32>, <32 x i32>* %arrayidx17, align 128
  %arrayidx18 = getelementptr inbounds <32 x i32>, <32 x i32>* %pkey.0542, i32 6
  %15 = load <32 x i32>, <32 x i32>* %arrayidx18, align 128
  %arrayidx19 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 12
  %16 = load <32 x i32>, <32 x i32>* %arrayidx19, align 128
  %arrayidx20 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 12
  %17 = load <32 x i32>, <32 x i32>* %arrayidx20, align 128
  %arrayidx22 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 14
  %18 = load <32 x i32>, <32 x i32>* %arrayidx22, align 128
  %arrayidx23 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 14
  %19 = load <32 x i32>, <32 x i32>* %arrayidx23, align 128
  %20 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %2, <32 x i32> %11)
  %21 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %20, <32 x i32> %11, <32 x i32> %2)
  %22 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %20, <32 x i32> %2, <32 x i32> %11)
  %23 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %20, <32 x i32> undef, <32 x i32> %3)
  %24 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %20, <32 x i32> %12, <32 x i32> undef)
  %25 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %7, <32 x i32> %15)
  %26 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %15, <32 x i32> %7)
  %27 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %7, <32 x i32> %15)
  %28 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %16, <32 x i32> %8)
  %29 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %8, <32 x i32> %16)
  %30 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %17, <32 x i32> %9)
  %31 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %25, <32 x i32> %9, <32 x i32> %17)
  %32 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %4, <32 x i32> %13)
  %33 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> %13, <32 x i32> %4)
  %34 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> %4, <32 x i32> %13)
  %35 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> undef, <32 x i32> %5)
  %36 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> %5, <32 x i32> undef)
  %37 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> %14, <32 x i32> %6)
  %38 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %32, <32 x i32> %6, <32 x i32> %14)
  %39 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> undef)
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> undef, <32 x i32> zeroinitializer)
  %41 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> %18, <32 x i32> %10)
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> %10, <32 x i32> %18)
  %43 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> %19, <32 x i32> undef)
  %44 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> zeroinitializer, <32 x i32> undef, <32 x i32> %19)
  %45 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %21, <32 x i32> %26)
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %26, <32 x i32> %21)
  %47 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %21, <32 x i32> %26)
  %48 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %28, <32 x i32> %23)
  %49 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %23, <32 x i32> %28)
  %50 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %30, <32 x i32> %24)
  %51 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %45, <32 x i32> %24, <32 x i32> %30)
  %52 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %22, <32 x i32> %27)
  %53 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %52, <32 x i32> %27, <32 x i32> %22)
  %54 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %52, <32 x i32> %22, <32 x i32> %27)
  %55 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %52, <32 x i32> %29, <32 x i32> undef)
  %56 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %52, <32 x i32> undef, <32 x i32> %31)
  %57 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %33, <32 x i32> %39)
  %58 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %57, <32 x i32> %39, <32 x i32> %33)
  %59 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %57, <32 x i32> %33, <32 x i32> %39)
  %60 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %57, <32 x i32> %41, <32 x i32> %35)
  %61 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %57, <32 x i32> %43, <32 x i32> %37)
  %62 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %34, <32 x i32> %40)
  %63 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %62, <32 x i32> %42, <32 x i32> %36)
  %64 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %62, <32 x i32> %38, <32 x i32> %44)
  %65 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %46, <32 x i32> %58)
  %66 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %65, <32 x i32> %58, <32 x i32> %46)
  %67 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %65, <32 x i32> %60, <32 x i32> %48)
  %68 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %65, <32 x i32> %61, <32 x i32> %50)
  %69 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %47, <32 x i32> %59)
  %70 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %69, <32 x i32> %51, <32 x i32> zeroinitializer)
  %71 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %53, <32 x i32> zeroinitializer)
  %72 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %71, <32 x i32> %63, <32 x i32> %55)
  %73 = tail call <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32> %54, <32 x i32> undef)
  %74 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1> %73, <32 x i32> %56, <32 x i32> %64)
  %75 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %68, <32 x i32> %67)
  %76 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %70, <32 x i32> undef)
  %77 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> zeroinitializer, <32 x i32> %72)
  %78 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %74, <32 x i32> zeroinitializer)
  %79 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %dAccum0.0539, <32 x i32> %75, i32 65537)
  %80 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %79, <32 x i32> zeroinitializer, i32 65537)
  %81 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %80, <32 x i32> zeroinitializer, i32 65537)
  %82 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %81, <32 x i32> %76, i32 65537)
  %83 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %82, <32 x i32> %77, i32 65537)
  %84 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %83, <32 x i32> zeroinitializer, i32 65537)
  %85 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %84, <32 x i32> undef, i32 65537)
  %86 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %85, <32 x i32> %78, i32 65537)
  store <32 x i32> %66, <32 x i32>* %pkey.0542, align 128
  store <32 x i32> %75, <32 x i32>* %pdata0.0541, align 128
  store <32 x i32> zeroinitializer, <32 x i32>* %arrayidx4, align 128
  store <32 x i32> zeroinitializer, <32 x i32>* undef, align 128
  store <32 x i32> zeroinitializer, <32 x i32>* %arrayidx20, align 128
  store <32 x i32> zeroinitializer, <32 x i32>* null, align 128
  %add.ptr48 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata0.0541, i32 16
  %add.ptr49 = getelementptr inbounds <32 x i32>, <32 x i32>* %pdata1.0540, i32 16
  br i1 false, label %for.end, label %for.body

for.end:
  %87 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %86)
  ret void
}

declare <1024 x i1> @llvm.hexagon.V6.vgtb.128B(<32 x i32>, <32 x i32>) #1

declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<1024 x i1>, <32 x i32>, <32 x i32>) #1

declare <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32>, <32 x i32>) #1

declare <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32>, <32 x i32>, i32) #1

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Check that store is post-incremented.
; CHECK: r{{[0-9]+:[0-9]+}} = vasrw(r{{[0-9]+:[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+:[0-9]+}} = vaslw(r{{[0-9]+:[0-9]+}},r{{[0-9]+}})
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(ptr nocapture %buf, ptr nocapture %dest, i32 %offset, i32 %oddBlock, i32 %gb) #0 {
entry:
  %0 = load i32, ptr %buf, align 4, !tbaa !0
  %shr = ashr i32 %0, %gb
  store i32 %shr, ptr %buf, align 4, !tbaa !0
  %not.tobool = icmp eq i32 %oddBlock, 0
  %1 = sub i32 %offset, %oddBlock
  %2 = zext i1 %not.tobool to i32
  %3 = and i32 %1, 7
  %4 = add i32 %2, %3
  %5 = add i32 %4, 8
  %p_sub8 = sub nsw i32 31, %gb
  %6 = insertelement <2 x i32> undef, i32 %p_sub8, i32 0
  %7 = insertelement <2 x i32> %6, i32 %p_sub8, i32 1
  %8 = bitcast <2 x i32> %7 to i64
  %9 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %8, i32 1)
  %10 = bitcast i64 %9 to <2 x i32>
  %11 = tail call i64 @llvm.hexagon.A2.combinew(i32 -1, i32 -1)
  %12 = bitcast i64 %11 to <2 x i32>
  %sub12p_vec = add <2 x i32> %10, %12
  %p_22 = add i32 %4, 64
  %p_d.018 = getelementptr i32, ptr %dest, i32 %4
  %p_d.01823 = getelementptr i32, ptr %dest, i32 %p_22
  %p_25 = add i32 %4, 72
  %p_arrayidx14 = getelementptr i32, ptr %dest, i32 %5
  %p_arrayidx1426 = getelementptr i32, ptr %dest, i32 %p_25
  %_p_scalar_ = load i32, ptr %p_d.018, align 4
  %_p_vec_ = insertelement <2 x i32> undef, i32 %_p_scalar_, i32 0
  %_p_scalar_27 = load i32, ptr %p_d.01823, align 4
  %_p_vec_28 = insertelement <2 x i32> %_p_vec_, i32 %_p_scalar_27, i32 1
  %13 = bitcast <2 x i32> %_p_vec_28 to i64
  %14 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %13, i32 31)
  %15 = bitcast i64 %14 to <2 x i32>
  %shr9p_vec = ashr <2 x i32> %_p_vec_28, %7
  %xorp_vec = xor <2 x i32> %15, %sub12p_vec
  %16 = bitcast <2 x i32> %shr9p_vec to i64
  %17 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %14, i64 %16)
  %18 = bitcast <2 x i32> %xorp_vec to i64
  %19 = tail call i64 @llvm.hexagon.C2.vmux(i32 %17, i64 %13, i64 %18)
  %20 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %19, i32 %gb)
  %21 = bitcast i64 %20 to <2 x i32>
  %22 = extractelement <2 x i32> %21, i32 0
  store i32 %22, ptr %p_arrayidx14, align 4
  %23 = extractelement <2 x i32> %21, i32 1
  store i32 %23, ptr %p_arrayidx1426, align 4
  store i32 %22, ptr %p_d.018, align 4
  store i32 %23, ptr %p_d.01823, align 4
  %p_21.1 = add i32 %4, 128
  %p_22.1 = add i32 %4, 192
  %p_d.018.1 = getelementptr i32, ptr %dest, i32 %p_21.1
  %p_d.01823.1 = getelementptr i32, ptr %dest, i32 %p_22.1
  %p_24.1 = add i32 %4, 136
  %p_25.1 = add i32 %4, 200
  %p_arrayidx14.1 = getelementptr i32, ptr %dest, i32 %p_24.1
  %p_arrayidx1426.1 = getelementptr i32, ptr %dest, i32 %p_25.1
  %_p_scalar_.1 = load i32, ptr %p_d.018.1, align 4
  %_p_vec_.1 = insertelement <2 x i32> undef, i32 %_p_scalar_.1, i32 0
  %_p_scalar_27.1 = load i32, ptr %p_d.01823.1, align 4
  %_p_vec_28.1 = insertelement <2 x i32> %_p_vec_.1, i32 %_p_scalar_27.1, i32 1
  %24 = bitcast <2 x i32> %_p_vec_28.1 to i64
  %25 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %24, i32 31)
  %26 = bitcast i64 %25 to <2 x i32>
  %shr9p_vec.1 = ashr <2 x i32> %_p_vec_28.1, %7
  %xorp_vec.1 = xor <2 x i32> %26, %sub12p_vec
  %27 = bitcast <2 x i32> %shr9p_vec.1 to i64
  %28 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %25, i64 %27)
  %29 = bitcast <2 x i32> %xorp_vec.1 to i64
  %30 = tail call i64 @llvm.hexagon.C2.vmux(i32 %28, i64 %24, i64 %29)
  %31 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %30, i32 %gb)
  %32 = bitcast i64 %31 to <2 x i32>
  %33 = extractelement <2 x i32> %32, i32 0
  store i32 %33, ptr %p_arrayidx14.1, align 4
  %34 = extractelement <2 x i32> %32, i32 1
  store i32 %34, ptr %p_arrayidx1426.1, align 4
  store i32 %33, ptr %p_d.018.1, align 4
  store i32 %34, ptr %p_d.01823.1, align 4
  %p_21.2 = add i32 %4, 256
  %p_22.2 = add i32 %4, 320
  %p_d.018.2 = getelementptr i32, ptr %dest, i32 %p_21.2
  %p_d.01823.2 = getelementptr i32, ptr %dest, i32 %p_22.2
  %p_24.2 = add i32 %4, 264
  %p_25.2 = add i32 %4, 328
  %p_arrayidx14.2 = getelementptr i32, ptr %dest, i32 %p_24.2
  %p_arrayidx1426.2 = getelementptr i32, ptr %dest, i32 %p_25.2
  %_p_scalar_.2 = load i32, ptr %p_d.018.2, align 4
  %_p_vec_.2 = insertelement <2 x i32> undef, i32 %_p_scalar_.2, i32 0
  %_p_scalar_27.2 = load i32, ptr %p_d.01823.2, align 4
  %_p_vec_28.2 = insertelement <2 x i32> %_p_vec_.2, i32 %_p_scalar_27.2, i32 1
  %35 = bitcast <2 x i32> %_p_vec_28.2 to i64
  %36 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %35, i32 31)
  %37 = bitcast i64 %36 to <2 x i32>
  %shr9p_vec.2 = ashr <2 x i32> %_p_vec_28.2, %7
  %xorp_vec.2 = xor <2 x i32> %37, %sub12p_vec
  %38 = bitcast <2 x i32> %shr9p_vec.2 to i64
  %39 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %36, i64 %38)
  %40 = bitcast <2 x i32> %xorp_vec.2 to i64
  %41 = tail call i64 @llvm.hexagon.C2.vmux(i32 %39, i64 %35, i64 %40)
  %42 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %41, i32 %gb)
  %43 = bitcast i64 %42 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  store i32 %44, ptr %p_arrayidx14.2, align 4
  %45 = extractelement <2 x i32> %43, i32 1
  store i32 %45, ptr %p_arrayidx1426.2, align 4
  store i32 %44, ptr %p_d.018.2, align 4
  store i32 %45, ptr %p_d.01823.2, align 4
  %p_21.3 = add i32 %4, 384
  %p_22.3 = add i32 %4, 448
  %p_d.018.3 = getelementptr i32, ptr %dest, i32 %p_21.3
  %p_d.01823.3 = getelementptr i32, ptr %dest, i32 %p_22.3
  %p_24.3 = add i32 %4, 392
  %p_25.3 = add i32 %4, 456
  %p_arrayidx14.3 = getelementptr i32, ptr %dest, i32 %p_24.3
  %p_arrayidx1426.3 = getelementptr i32, ptr %dest, i32 %p_25.3
  %_p_scalar_.3 = load i32, ptr %p_d.018.3, align 4
  %_p_vec_.3 = insertelement <2 x i32> undef, i32 %_p_scalar_.3, i32 0
  %_p_scalar_27.3 = load i32, ptr %p_d.01823.3, align 4
  %_p_vec_28.3 = insertelement <2 x i32> %_p_vec_.3, i32 %_p_scalar_27.3, i32 1
  %46 = bitcast <2 x i32> %_p_vec_28.3 to i64
  %47 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %46, i32 31)
  %48 = bitcast i64 %47 to <2 x i32>
  %shr9p_vec.3 = ashr <2 x i32> %_p_vec_28.3, %7
  %xorp_vec.3 = xor <2 x i32> %48, %sub12p_vec
  %49 = bitcast <2 x i32> %shr9p_vec.3 to i64
  %50 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %47, i64 %49)
  %51 = bitcast <2 x i32> %xorp_vec.3 to i64
  %52 = tail call i64 @llvm.hexagon.C2.vmux(i32 %50, i64 %46, i64 %51)
  %53 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %52, i32 %gb)
  %54 = bitcast i64 %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i32 0
  store i32 %55, ptr %p_arrayidx14.3, align 4
  %56 = extractelement <2 x i32> %54, i32 1
  store i32 %56, ptr %p_arrayidx1426.3, align 4
  store i32 %55, ptr %p_d.018.3, align 4
  store i32 %56, ptr %p_d.01823.3, align 4
  %p_21.4 = add i32 %4, 512
  %p_22.4 = add i32 %4, 576
  %p_d.018.4 = getelementptr i32, ptr %dest, i32 %p_21.4
  %p_d.01823.4 = getelementptr i32, ptr %dest, i32 %p_22.4
  %p_24.4 = add i32 %4, 520
  %p_25.4 = add i32 %4, 584
  %p_arrayidx14.4 = getelementptr i32, ptr %dest, i32 %p_24.4
  %p_arrayidx1426.4 = getelementptr i32, ptr %dest, i32 %p_25.4
  %_p_scalar_.4 = load i32, ptr %p_d.018.4, align 4
  %_p_vec_.4 = insertelement <2 x i32> undef, i32 %_p_scalar_.4, i32 0
  %_p_scalar_27.4 = load i32, ptr %p_d.01823.4, align 4
  %_p_vec_28.4 = insertelement <2 x i32> %_p_vec_.4, i32 %_p_scalar_27.4, i32 1
  %57 = bitcast <2 x i32> %_p_vec_28.4 to i64
  %58 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %57, i32 31)
  %59 = bitcast i64 %58 to <2 x i32>
  %shr9p_vec.4 = ashr <2 x i32> %_p_vec_28.4, %7
  %xorp_vec.4 = xor <2 x i32> %59, %sub12p_vec
  %60 = bitcast <2 x i32> %shr9p_vec.4 to i64
  %61 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %58, i64 %60)
  %62 = bitcast <2 x i32> %xorp_vec.4 to i64
  %63 = tail call i64 @llvm.hexagon.C2.vmux(i32 %61, i64 %57, i64 %62)
  %64 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %63, i32 %gb)
  %65 = bitcast i64 %64 to <2 x i32>
  %66 = extractelement <2 x i32> %65, i32 0
  store i32 %66, ptr %p_arrayidx14.4, align 4
  %67 = extractelement <2 x i32> %65, i32 1
  store i32 %67, ptr %p_arrayidx1426.4, align 4
  store i32 %66, ptr %p_d.018.4, align 4
  store i32 %67, ptr %p_d.01823.4, align 4
  %p_21.5 = add i32 %4, 640
  %p_22.5 = add i32 %4, 704
  %p_d.018.5 = getelementptr i32, ptr %dest, i32 %p_21.5
  %p_d.01823.5 = getelementptr i32, ptr %dest, i32 %p_22.5
  %p_24.5 = add i32 %4, 648
  %p_25.5 = add i32 %4, 712
  %p_arrayidx14.5 = getelementptr i32, ptr %dest, i32 %p_24.5
  %p_arrayidx1426.5 = getelementptr i32, ptr %dest, i32 %p_25.5
  %_p_scalar_.5 = load i32, ptr %p_d.018.5, align 4
  %_p_vec_.5 = insertelement <2 x i32> undef, i32 %_p_scalar_.5, i32 0
  %_p_scalar_27.5 = load i32, ptr %p_d.01823.5, align 4
  %_p_vec_28.5 = insertelement <2 x i32> %_p_vec_.5, i32 %_p_scalar_27.5, i32 1
  %68 = bitcast <2 x i32> %_p_vec_28.5 to i64
  %69 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %68, i32 31)
  %70 = bitcast i64 %69 to <2 x i32>
  %shr9p_vec.5 = ashr <2 x i32> %_p_vec_28.5, %7
  %xorp_vec.5 = xor <2 x i32> %70, %sub12p_vec
  %71 = bitcast <2 x i32> %shr9p_vec.5 to i64
  %72 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %69, i64 %71)
  %73 = bitcast <2 x i32> %xorp_vec.5 to i64
  %74 = tail call i64 @llvm.hexagon.C2.vmux(i32 %72, i64 %68, i64 %73)
  %75 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %74, i32 %gb)
  %76 = bitcast i64 %75 to <2 x i32>
  %77 = extractelement <2 x i32> %76, i32 0
  store i32 %77, ptr %p_arrayidx14.5, align 4
  %78 = extractelement <2 x i32> %76, i32 1
  store i32 %78, ptr %p_arrayidx1426.5, align 4
  store i32 %77, ptr %p_d.018.5, align 4
  store i32 %78, ptr %p_d.01823.5, align 4
  %p_21.6 = add i32 %4, 768
  %p_22.6 = add i32 %4, 832
  %p_d.018.6 = getelementptr i32, ptr %dest, i32 %p_21.6
  %p_d.01823.6 = getelementptr i32, ptr %dest, i32 %p_22.6
  %p_24.6 = add i32 %4, 776
  %p_25.6 = add i32 %4, 840
  %p_arrayidx14.6 = getelementptr i32, ptr %dest, i32 %p_24.6
  %p_arrayidx1426.6 = getelementptr i32, ptr %dest, i32 %p_25.6
  %_p_scalar_.6 = load i32, ptr %p_d.018.6, align 4
  %_p_vec_.6 = insertelement <2 x i32> undef, i32 %_p_scalar_.6, i32 0
  %_p_scalar_27.6 = load i32, ptr %p_d.01823.6, align 4
  %_p_vec_28.6 = insertelement <2 x i32> %_p_vec_.6, i32 %_p_scalar_27.6, i32 1
  %79 = bitcast <2 x i32> %_p_vec_28.6 to i64
  %80 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %79, i32 31)
  %81 = bitcast i64 %80 to <2 x i32>
  %shr9p_vec.6 = ashr <2 x i32> %_p_vec_28.6, %7
  %xorp_vec.6 = xor <2 x i32> %81, %sub12p_vec
  %82 = bitcast <2 x i32> %shr9p_vec.6 to i64
  %83 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %80, i64 %82)
  %84 = bitcast <2 x i32> %xorp_vec.6 to i64
  %85 = tail call i64 @llvm.hexagon.C2.vmux(i32 %83, i64 %79, i64 %84)
  %86 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %85, i32 %gb)
  %87 = bitcast i64 %86 to <2 x i32>
  %88 = extractelement <2 x i32> %87, i32 0
  store i32 %88, ptr %p_arrayidx14.6, align 4
  %89 = extractelement <2 x i32> %87, i32 1
  store i32 %89, ptr %p_arrayidx1426.6, align 4
  store i32 %88, ptr %p_d.018.6, align 4
  store i32 %89, ptr %p_d.01823.6, align 4
  %p_21.7 = add i32 %4, 896
  %p_22.7 = add i32 %4, 960
  %p_d.018.7 = getelementptr i32, ptr %dest, i32 %p_21.7
  %p_d.01823.7 = getelementptr i32, ptr %dest, i32 %p_22.7
  %p_24.7 = add i32 %4, 904
  %p_25.7 = add i32 %4, 968
  %p_arrayidx14.7 = getelementptr i32, ptr %dest, i32 %p_24.7
  %p_arrayidx1426.7 = getelementptr i32, ptr %dest, i32 %p_25.7
  %_p_scalar_.7 = load i32, ptr %p_d.018.7, align 4
  %_p_vec_.7 = insertelement <2 x i32> undef, i32 %_p_scalar_.7, i32 0
  %_p_scalar_27.7 = load i32, ptr %p_d.01823.7, align 4
  %_p_vec_28.7 = insertelement <2 x i32> %_p_vec_.7, i32 %_p_scalar_27.7, i32 1
  %90 = bitcast <2 x i32> %_p_vec_28.7 to i64
  %91 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %90, i32 31)
  %92 = bitcast i64 %91 to <2 x i32>
  %shr9p_vec.7 = ashr <2 x i32> %_p_vec_28.7, %7
  %xorp_vec.7 = xor <2 x i32> %92, %sub12p_vec
  %93 = bitcast <2 x i32> %shr9p_vec.7 to i64
  %94 = tail call i32 @llvm.hexagon.A2.vcmpweq(i64 %91, i64 %93)
  %95 = bitcast <2 x i32> %xorp_vec.7 to i64
  %96 = tail call i64 @llvm.hexagon.C2.vmux(i32 %94, i64 %90, i64 %95)
  %97 = tail call i64 @llvm.hexagon.S2.asl.r.vw(i64 %96, i32 %gb)
  %98 = bitcast i64 %97 to <2 x i32>
  %99 = extractelement <2 x i32> %98, i32 0
  store i32 %99, ptr %p_arrayidx14.7, align 4
  %100 = extractelement <2 x i32> %98, i32 1
  store i32 %100, ptr %p_arrayidx1426.7, align 4
  store i32 %99, ptr %p_d.018.7, align 4
  store i32 %100, ptr %p_d.01823.7, align 4
  ret void
}

declare i64 @llvm.hexagon.S2.asr.i.vw(i64, i32) #1

declare i64 @llvm.hexagon.S2.asl.i.vw(i64, i32) #1

declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

declare i32 @llvm.hexagon.A2.vcmpweq(i64, i64) #1

declare i64 @llvm.hexagon.C2.vmux(i32, i64, i64) #1

declare i64 @llvm.hexagon.S2.asl.r.vw(i64, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}

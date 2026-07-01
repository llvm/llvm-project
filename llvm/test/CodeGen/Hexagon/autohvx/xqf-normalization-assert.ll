; Tests if the test crashes due to generation of normalization instructions
; which do not have equivalent defs in the dominator basicblock.

; RUN:   llc -mtriple=hexagon-unknown-elf -O2 -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true -hexagon-qfloat-mode=lossy < %s -o /dev/null

@c0_coeffs_asin_vhf = internal unnamed_addr constant [32 x float] [float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xC0AE6428A0000000, float 0xC01AEBA200000000, float 0xBFDAF02840000000, float 0xBFA6CBD5A0000000, float 0xBF75153560000000, float 0xBF3EA129C0000000, float 0xBEF513C980000000, float 0xBE418295E0000000, float 0x3E1AE29C20000000, float 0x3EF4410BA0000000, float 0x3F3D8DCE60000000, float 0x3F749B72E0000000, float 0x3FA666F3E0000000, float 0x3FDA201EC0000000, float 0x40194BFD40000000, float 0x40ABE59AC0000000], align 128
@c1_coeffs_asin_vhf = internal unnamed_addr constant [32 x float] [float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xC0D0766D60000000, float 0xC04129FCE0000000, float 0xBFFB611840000000, float 0x3FE4688DA0000000, float 0x3FEE395540000000, float 0x3FEFC3F260000000, float 0x3FEFFB6980000000, float 0x3FEFFFFB60000000, float 0x3FEFFFFCE0000000, float 0x3FEFFB8EE0000000, float 0x3FEFC5B520000000, float 0x3FEE421260000000, float 0x3FE4960460000000, float 0xBFFA2F48C0000000, float 0xC040284FC0000000, float 0xC0CE476740000000], align 128
@c2_coeffs_asin_vhf = internal unnamed_addr constant [32 x float] [float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xC0DABF49E0000000, float 0xC0517AB3E0000000, float 0xC01A8AD340000000, float 0xBFF2146300000000, float 0xBFCDA870C0000000, float 0xBFA6FE7CC0000000, float 0xBF78FBAA20000000, float 0xBF20DAF2E0000000, float 0x3F1BC13500000000, float 0x3F785EEC80000000, float 0x3FA674DBC0000000, float 0x3FCD305400000000, float 0x3FF1D70400000000, float 0x4019E26E20000000, float 0x40508B4500000000, float 0x40D8A49620000000], align 128
@c3_coeffs_asin_vhf = internal unnamed_addr constant [32 x float] [float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xC0D34EAC80000000, float 0xC04EE8CDC0000000, float 0xC01CD2A460000000, float 0xBFF761B920000000, float 0xBFD2806420000000, float 0x3FA1A14760000000, float 0x3FC1343380000000, float 0x3FC4FB5360000000, float 0x3FC503B6E0000000, float 0x3FC14630A0000000, float 0x3FA2CA8A40000000, float 0xBFD224E9A0000000, float 0xBFF71832E0000000, float 0xC01C2DF8C0000000, float 0xC04D5D9600000000, float 0xC0D1D22280000000], align 128
@c4_coeffs_asin_vhf = internal unnamed_addr constant [32 x float] [float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xC0B4E6B460000000, float 0xC034E86040000000, float 0xC009BCE840000000, float 0xBFEEE8AF00000000, float 0xBFD8FF8A00000000, float 0xBFC6F58460000000, float 0xBFB4E58DC0000000, float 0xBF98DF4700000000, float 0x3F97E3BFA0000000, float 0x3FB4B4B500000000, float 0x3FC6B9A2A0000000, float 0x3FD8CB6B20000000, float 0x3FEEA6B940000000, float 0x4009444400000000, float 0x4033F3EE00000000, float 0x40B353BBC0000000], align 128

; Function Attrs: nounwind
define i32 @qhmath_hvx_asin_ahf(ptr noalias noundef %input, ptr noalias noundef %output, i32 noundef %size) local_unnamed_addr #0 {
entry:
  %and = and i32 %size, 63
  %mul = shl nuw nsw i32 %and, 1
  %cmp = icmp eq ptr %input, null
  %cmp1 = icmp eq ptr %output, null
  %or.cond = or i1 %cmp, %cmp1
  %cmp3 = icmp eq i32 %size, 0
  %or.cond46 = or i1 %or.cond, %cmp3
  br i1 %or.cond46, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %incdec.ptr = getelementptr inbounds <32 x i32>, ptr %input, i32 1
  %0 = load <32 x i32>, ptr %input, align 128
  %cmp4102 = icmp ugt i32 %size, 127
  br i1 %cmp4102, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %if.end
  %div82 = lshr i32 %size, 6
  %sub = add nsw i32 %div82, -1
  %1 = ptrtoint ptr %input to i32
  %2 = load <32 x i32>, ptr @c0_coeffs_asin_vhf, align 128
  %3 = load <32 x i32>, ptr @c1_coeffs_asin_vhf, align 128
  %4 = load <32 x i32>, ptr @c2_coeffs_asin_vhf, align 128
  %5 = load <32 x i32>, ptr @c3_coeffs_asin_vhf, align 128
  %6 = load <32 x i32>, ptr @c4_coeffs_asin_vhf, align 128
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body12, %if.end8
  %output_v_ptr.1.lcssa = phi ptr [ %output_v_ptr.0103, %if.end8 ], [ %incdec.ptr14, %for.body12 ]
  %input_v_ptr.1.lcssa = phi ptr [ %input_v_ptr.0104, %if.end8 ], [ %incdec.ptr13, %for.body12 ]
  %slinep.1.lcssa = phi <32 x i32> [ %slinep.0105, %if.end8 ], [ %37, %for.body12 ]
  %cmp4 = icmp sgt i32 %i.0106, 64
  br i1 %cmp4, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.loopexit, %if.end
  %output_v_ptr.0.lcssa = phi ptr [ %output, %if.end ], [ %output_v_ptr.1.lcssa, %for.cond.loopexit ]
  %input_v_ptr.0.lcssa = phi ptr [ %incdec.ptr, %if.end ], [ %input_v_ptr.1.lcssa, %for.cond.loopexit ]
  %slinep.0.lcssa = phi <32 x i32> [ %0, %if.end ], [ %slinep.1.lcssa, %for.cond.loopexit ]
  %cmp18.not = icmp ult i32 %size, 64
  br i1 %cmp18.not, label %if.end25, label %if.then19

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %i.0106 = phi i32 [ %sub, %for.body.lr.ph ], [ %sub5, %for.cond.loopexit ]
  %slinep.0105 = phi <32 x i32> [ %0, %for.body.lr.ph ], [ %slinep.1.lcssa, %for.cond.loopexit ]
  %input_v_ptr.0104 = phi ptr [ %incdec.ptr, %for.body.lr.ph ], [ %input_v_ptr.1.lcssa, %for.cond.loopexit ]
  %output_v_ptr.0103 = phi ptr [ %output, %for.body.lr.ph ], [ %output_v_ptr.1.lcssa, %for.cond.loopexit ]
  %7 = tail call i32 @llvm.hexagon.A2.min(i32 %i.0106, i32 64)
  %sub5 = add nsw i32 %i.0106, -64
  %8 = tail call i32 @llvm.hexagon.A2.min(i32 %sub5, i32 64)
  %cmp6 = icmp sgt i32 %8, 0
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %for.body
  %add.ptr = getelementptr inbounds <32 x i32>, ptr %input_v_ptr.0104, i32 64
  %conv.mask.i = and i32 %8, 65535
  %_HEXAGON_V64_internal_union.sroa.0.0.insert.ext.i = zext i32 %conv.mask.i to i64
  %_HEXAGON_V64_internal_union.sroa.0.0.insert.insert.i = or i64 %_HEXAGON_V64_internal_union.sroa.0.0.insert.ext.i, 549764202496
  tail call void asm sideeffect " l2fetch($0,$1) ", "r,r"(ptr nonnull %add.ptr, i64 %_HEXAGON_V64_internal_union.sroa.0.0.insert.insert.i) #3
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %for.body
  %cmp1095 = icmp sgt i32 %7, 0
  br i1 %cmp1095, label %for.body12.lr.ph, label %for.cond.loopexit

for.body12.lr.ph:                                 ; preds = %if.end8
  %9 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 18430)
  %10 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15360)
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B()
  %12 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15)
  %13 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 4112)
  %14 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 19456)
  %15 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 48128)
  %16 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %9, <32 x i32> %11)
  %17 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %2, <32 x i32> %11)
  %18 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %3, <32 x i32> %11)
  %19 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %4, <32 x i32> %11)
  %20 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %5, <32 x i32> %11)
  %21 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %6, <32 x i32> %11)
  %22 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %17)
  %23 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %18)
  %24 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %19)
  %25 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %20)
  %26 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %21)
  %27 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %22)
  %28 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %22)
  %29 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %23)
  %30 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %23)
  %31 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %24)
  %32 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %24)
  %33 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %25)
  %34 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %25)
  %35 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %26)
  %36 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %26)
  br label %for.body12

for.body12:                                       ; preds = %for.body12, %for.body12.lr.ph
  %j.099 = phi i32 [ 0, %for.body12.lr.ph ], [ %inc, %for.body12 ]
  %slinep.198 = phi <32 x i32> [ %slinep.0105, %for.body12.lr.ph ], [ %37, %for.body12 ]
  %input_v_ptr.197 = phi ptr [ %input_v_ptr.0104, %for.body12.lr.ph ], [ %incdec.ptr13, %for.body12 ]
  %output_v_ptr.196 = phi ptr [ %output_v_ptr.0103, %for.body12.lr.ph ], [ %incdec.ptr14, %for.body12 ]
  %incdec.ptr13 = getelementptr inbounds <32 x i32>, ptr %input_v_ptr.197, i32 1
  %37 = load <32 x i32>, ptr %input_v_ptr.197, align 128
  %38 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %37, <32 x i32> %slinep.198, i32 %1)
  %39 = tail call <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32> %38)
  %40 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %39, <32 x i32> %15)
  %41 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32> %40, <32 x i32> %16)
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %41, <32 x i32> %14)
  %43 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %42)
  %44 = tail call <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32> %43, i32 6)
  %45 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %44, <32 x i32> %12)
  %46 = tail call <32 x i32> @llvm.hexagon.V6.vshuffb.128B(<32 x i32> %45)
  %47 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> %46, <32 x i32> %13)
  %48 = tail call <32 x i32> @llvm.hexagon.V6.vaslw.128B(<32 x i32> %47, i32 16)
  %49 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %47, <32 x i32> %27, i32 1)
  %50 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %49, <32 x i32> %48, <32 x i32> %28, i32 1)
  %51 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %47, <32 x i32> %29, i32 1)
  %52 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %51, <32 x i32> %48, <32 x i32> %30, i32 1)
  %53 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %47, <32 x i32> %31, i32 1)
  %54 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %53, <32 x i32> %48, <32 x i32> %32, i32 1)
  %55 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %47, <32 x i32> %33, i32 1)
  %56 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %55, <32 x i32> %48, <32 x i32> %34, i32 1)
  %57 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %47, <32 x i32> %35, i32 1)
  %58 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %57, <32 x i32> %48, <32 x i32> %36, i32 1)
  %59 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32> %38, <32 x i32> %10)
  %60 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %58)
  %61 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %59)
  %62 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %60, <32 x i32> %61)
  %63 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %56)
  %64 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %62, <32 x i32> %63)
  %65 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %64, <32 x i32> %61)
  %66 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %54)
  %67 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %65, <32 x i32> %66)
  %68 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %67, <32 x i32> %61)
  %69 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %52)
  %70 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %68, <32 x i32> %69)
  %71 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %70, <32 x i32> %61)
  %72 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %50)
  %73 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %71, <32 x i32> %72)
  %output_dv.sroa.0.0.vecblend84.i.i = shufflevector <32 x i32> %73, <32 x i32> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %74 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %58)
  %75 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %59)
  %76 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %74, <32 x i32> %75)
  %77 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %56)
  %78 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %76, <32 x i32> %77)
  %79 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %78, <32 x i32> %75)
  %80 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %54)
  %81 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %79, <32 x i32> %80)
  %82 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %81, <32 x i32> %75)
  %83 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %52)
  %84 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %82, <32 x i32> %83)
  %85 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %84, <32 x i32> %75)
  %86 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %50)
  %87 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %85, <32 x i32> %86)
  %output_dv.sroa.0.128.vec.expand117.i.i = shufflevector <32 x i32> %87, <32 x i32> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %output_dv.sroa.0.128.vecblend118.i.i = shufflevector <64 x i32> %output_dv.sroa.0.0.vecblend84.i.i, <64 x i32> %output_dv.sroa.0.128.vec.expand117.i.i, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  %88 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %output_dv.sroa.0.128.vecblend118.i.i)
  %incdec.ptr14 = getelementptr inbounds <32 x i32>, ptr %output_v_ptr.196, i32 1
  store <32 x i32> %88, ptr %output_v_ptr.196, align 4
  %inc = add nuw nsw i32 %j.099, 1
  %exitcond.not = icmp eq i32 %inc, %7
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body12

if.then19:                                        ; preds = %for.cond.cleanup
  %89 = ptrtoint ptr %input_v_ptr.0.lcssa to i32
  %and.i = and i32 %89, 127
  %90 = or i32 %and.i, %and
  %or.cond47 = icmp eq i32 %90, 0
  br i1 %or.cond47, label %cond.end, label %cond.false

cond.false:                                       ; preds = %if.then19
  %incdec.ptr22 = getelementptr inbounds <32 x i32>, ptr %input_v_ptr.0.lcssa, i32 1
  %91 = load <32 x i32>, ptr %input_v_ptr.0.lcssa, align 128
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %if.then19
  %input_v_ptr.2 = phi ptr [ %incdec.ptr22, %cond.false ], [ %input_v_ptr.0.lcssa, %if.then19 ]
  %cond = phi <32 x i32> [ %91, %cond.false ], [ %slinep.0.lcssa, %if.then19 ]
  %92 = ptrtoint ptr %input to i32
  %93 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %cond, <32 x i32> %slinep.0.lcssa, i32 %92)
  %94 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 18430)
  %95 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15360)
  %96 = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B()
  %97 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15)
  %98 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 4112)
  %99 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 19456)
  %100 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 48128)
  %101 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %94, <32 x i32> %96)
  %102 = load <32 x i32>, ptr @c0_coeffs_asin_vhf, align 128
  %103 = load <32 x i32>, ptr @c1_coeffs_asin_vhf, align 128
  %104 = load <32 x i32>, ptr @c2_coeffs_asin_vhf, align 128
  %105 = load <32 x i32>, ptr @c3_coeffs_asin_vhf, align 128
  %106 = load <32 x i32>, ptr @c4_coeffs_asin_vhf, align 128
  %107 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %102, <32 x i32> %96)
  %108 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %103, <32 x i32> %96)
  %109 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %104, <32 x i32> %96)
  %110 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %105, <32 x i32> %96)
  %111 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %106, <32 x i32> %96)
  %112 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %107)
  %113 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %108)
  %114 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %109)
  %115 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %110)
  %116 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %111)
  %117 = tail call <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32> %93)
  %118 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %117, <32 x i32> %100)
  %119 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32> %118, <32 x i32> %101)
  %120 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %119, <32 x i32> %99)
  %121 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %120)
  %122 = tail call <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32> %121, i32 6)
  %123 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %122, <32 x i32> %97)
  %124 = tail call <32 x i32> @llvm.hexagon.V6.vshuffb.128B(<32 x i32> %123)
  %125 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> %124, <32 x i32> %98)
  %126 = tail call <32 x i32> @llvm.hexagon.V6.vaslw.128B(<32 x i32> %125, i32 16)
  %127 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %112)
  %128 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %125, <32 x i32> %127, i32 1)
  %129 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %112)
  %130 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %128, <32 x i32> %126, <32 x i32> %129, i32 1)
  %131 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %113)
  %132 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %125, <32 x i32> %131, i32 1)
  %133 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %113)
  %134 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %132, <32 x i32> %126, <32 x i32> %133, i32 1)
  %135 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %114)
  %136 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %125, <32 x i32> %135, i32 1)
  %137 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %114)
  %138 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %136, <32 x i32> %126, <32 x i32> %137, i32 1)
  %139 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %115)
  %140 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %125, <32 x i32> %139, i32 1)
  %141 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %115)
  %142 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %140, <32 x i32> %126, <32 x i32> %141, i32 1)
  %143 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %116)
  %144 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %125, <32 x i32> %143, i32 1)
  %145 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %116)
  %146 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %144, <32 x i32> %126, <32 x i32> %145, i32 1)
  %147 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32> %93, <32 x i32> %95)
  %148 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %146)
  %149 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %147)
  %150 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %148, <32 x i32> %149)
  %151 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %142)
  %152 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %150, <32 x i32> %151)
  %153 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %152, <32 x i32> %149)
  %154 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %138)
  %155 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %153, <32 x i32> %154)
  %156 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %155, <32 x i32> %149)
  %157 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %134)
  %158 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %156, <32 x i32> %157)
  %159 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %158, <32 x i32> %149)
  %160 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %130)
  %161 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %159, <32 x i32> %160)
  %output_dv.sroa.0.0.vecblend84.i.i83 = shufflevector <32 x i32> %161, <32 x i32> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %162 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %146)
  %163 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %147)
  %164 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %162, <32 x i32> %163)
  %165 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %142)
  %166 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %164, <32 x i32> %165)
  %167 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %166, <32 x i32> %163)
  %168 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %138)
  %169 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %167, <32 x i32> %168)
  %170 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %169, <32 x i32> %163)
  %171 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %134)
  %172 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %170, <32 x i32> %171)
  %173 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %172, <32 x i32> %163)
  %174 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %130)
  %175 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %173, <32 x i32> %174)
  %output_dv.sroa.0.128.vec.expand117.i.i84 = shufflevector <32 x i32> %175, <32 x i32> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %output_dv.sroa.0.128.vecblend118.i.i85 = shufflevector <64 x i32> %output_dv.sroa.0.0.vecblend84.i.i83, <64 x i32> %output_dv.sroa.0.128.vec.expand117.i.i84, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  %176 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %output_dv.sroa.0.128.vecblend118.i.i85)
  %incdec.ptr24 = getelementptr inbounds <32 x i32>, ptr %output_v_ptr.0.lcssa, i32 1
  store <32 x i32> %176, ptr %output_v_ptr.0.lcssa, align 4
  br label %if.end25

if.end25:                                         ; preds = %cond.end, %for.cond.cleanup
  %output_v_ptr.2 = phi ptr [ %incdec.ptr24, %cond.end ], [ %output_v_ptr.0.lcssa, %for.cond.cleanup ]
  %input_v_ptr.3 = phi ptr [ %input_v_ptr.2, %cond.end ], [ %input_v_ptr.0.lcssa, %for.cond.cleanup ]
  %slinep.2 = phi <32 x i32> [ %cond, %cond.end ], [ %slinep.0.lcssa, %for.cond.cleanup ]
  %cmp26.not = icmp eq i32 %and, 0
  br i1 %cmp26.not, label %cleanup, label %if.then27

if.then27:                                        ; preds = %if.end25
  %177 = ptrtoint ptr %input_v_ptr.3 to i32
  %and.i86 = and i32 %177, 127
  %add.i = add nuw nsw i32 %and.i86, %mul
  %cmp.i87 = icmp ugt i32 %add.i, 128
  br i1 %cmp.i87, label %cond.false31, label %cond.end33

cond.false31:                                     ; preds = %if.then27
  %178 = load <32 x i32>, ptr %input_v_ptr.3, align 128
  br label %cond.end33

cond.end33:                                       ; preds = %cond.false31, %if.then27
  %cond34 = phi <32 x i32> [ %178, %cond.false31 ], [ %slinep.2, %if.then27 ]
  %179 = ptrtoint ptr %input to i32
  %180 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %cond34, <32 x i32> %slinep.2, i32 %179)
  %181 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 18430)
  %182 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15360)
  %183 = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B()
  %184 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15)
  %185 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 4112)
  %186 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 19456)
  %187 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 48128)
  %188 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %181, <32 x i32> %183)
  %189 = load <32 x i32>, ptr @c0_coeffs_asin_vhf, align 128
  %190 = load <32 x i32>, ptr @c1_coeffs_asin_vhf, align 128
  %191 = load <32 x i32>, ptr @c2_coeffs_asin_vhf, align 128
  %192 = load <32 x i32>, ptr @c3_coeffs_asin_vhf, align 128
  %193 = load <32 x i32>, ptr @c4_coeffs_asin_vhf, align 128
  %194 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %189, <32 x i32> %183)
  %195 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %190, <32 x i32> %183)
  %196 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %191, <32 x i32> %183)
  %197 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %192, <32 x i32> %183)
  %198 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %193, <32 x i32> %183)
  %199 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %194)
  %200 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %195)
  %201 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %196)
  %202 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %197)
  %203 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %198)
  %204 = tail call <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32> %180)
  %205 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %204, <32 x i32> %187)
  %206 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32> %205, <32 x i32> %188)
  %207 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %206, <32 x i32> %186)
  %208 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %207)
  %209 = tail call <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32> %208, i32 6)
  %210 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %209, <32 x i32> %184)
  %211 = tail call <32 x i32> @llvm.hexagon.V6.vshuffb.128B(<32 x i32> %210)
  %212 = tail call <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32> %211, <32 x i32> %185)
  %213 = tail call <32 x i32> @llvm.hexagon.V6.vaslw.128B(<32 x i32> %212, i32 16)
  %214 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %199)
  %215 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %212, <32 x i32> %214, i32 1)
  %216 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %199)
  %217 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %215, <32 x i32> %213, <32 x i32> %216, i32 1)
  %218 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %200)
  %219 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %212, <32 x i32> %218, i32 1)
  %220 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %200)
  %221 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %219, <32 x i32> %213, <32 x i32> %220, i32 1)
  %222 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %201)
  %223 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %212, <32 x i32> %222, i32 1)
  %224 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %201)
  %225 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %223, <32 x i32> %213, <32 x i32> %224, i32 1)
  %226 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %202)
  %227 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %212, <32 x i32> %226, i32 1)
  %228 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %202)
  %229 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %227, <32 x i32> %213, <32 x i32> %228, i32 1)
  %230 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %203)
  %231 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> %212, <32 x i32> %230, i32 1)
  %232 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %203)
  %233 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %231, <32 x i32> %213, <32 x i32> %232, i32 1)
  %234 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32> %180, <32 x i32> %182)
  %235 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %233)
  %236 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %234)
  %237 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %235, <32 x i32> %236)
  %238 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %229)
  %239 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %237, <32 x i32> %238)
  %240 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %239, <32 x i32> %236)
  %241 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %225)
  %242 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %240, <32 x i32> %241)
  %243 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %242, <32 x i32> %236)
  %244 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %221)
  %245 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %243, <32 x i32> %244)
  %246 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %245, <32 x i32> %236)
  %247 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %217)
  %248 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %246, <32 x i32> %247)
  %output_dv.sroa.0.0.vecblend84.i.i89 = shufflevector <32 x i32> %248, <32 x i32> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %249 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %233)
  %250 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %234)
  %251 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %249, <32 x i32> %250)
  %252 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %229)
  %253 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %251, <32 x i32> %252)
  %254 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %253, <32 x i32> %250)
  %255 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %225)
  %256 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %254, <32 x i32> %255)
  %257 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %256, <32 x i32> %250)
  %258 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %221)
  %259 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %257, <32 x i32> %258)
  %260 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %259, <32 x i32> %250)
  %261 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %217)
  %262 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %260, <32 x i32> %261)
  %output_dv.sroa.0.128.vec.expand117.i.i90 = shufflevector <32 x i32> %262, <32 x i32> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %output_dv.sroa.0.128.vecblend118.i.i91 = shufflevector <64 x i32> %output_dv.sroa.0.0.vecblend84.i.i89, <64 x i32> %output_dv.sroa.0.128.vec.expand117.i.i90, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  %263 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %output_dv.sroa.0.128.vecblend118.i.i91)
  %264 = ptrtoint ptr %output_v_ptr.2 to i32
  %265 = tail call <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32> %263, <32 x i32> %263, i32 %264)
  %and.i92 = and i32 %264, 127
  %add.i93 = add nuw nsw i32 %and.i92, %mul
  %266 = tail call <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(i32 %add.i93)
  %267 = tail call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %266, i32 -1)
  %cmp.i94 = icmp ugt i32 %add.i93, 128
  br i1 %cmp.i94, label %if.then.i, label %vstu_variable.exit

if.then.i:                                        ; preds = %cond.end33
  %add.ptr.i = getelementptr inbounds <32 x i32>, ptr %output_v_ptr.2, i32 1
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1> %266, ptr nonnull %add.ptr.i, <32 x i32> %265)
  %268 = tail call <128 x i1> @llvm.hexagon.V6.veqb.128B(<32 x i32> %265, <32 x i32> %265)
  %269 = tail call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %268, i32 -1)
  br label %vstu_variable.exit

vstu_variable.exit:                               ; preds = %if.then.i, %cond.end33
  %qr.0.i = phi <32 x i32> [ %269, %if.then.i ], [ %267, %cond.end33 ]
  %270 = tail call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %264)
  %271 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %qr.0.i, i32 -1)
  %272 = tail call <128 x i1> @llvm.hexagon.V6.pred.or.n.128B(<128 x i1> %270, <128 x i1> %271)
  tail call void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(<128 x i1> %272, ptr %output_v_ptr.2, <32 x i32> %265)
  br label %cleanup

cleanup:                                          ; preds = %vstu_variable.exit, %if.end25, %entry
  %retval.0 = phi i32 [ -1, %entry ], [ 0, %vstu_variable.exit ], [ 0, %if.end25 ]
  ret i32 %retval.0
}

declare i32 @llvm.hexagon.A2.min(i32, i32) #1
declare <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vlsrh.128B(<32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vshuffb.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vor.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaslw.128B(<32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32>, <32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(i32) #1
declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #2
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.veqb.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.or.n.128B(<128 x i1>, <128 x i1>) #1
declare void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(<128 x i1>, ptr, <32 x i32>) #2

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvxv79,+v79,-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(write) }
attributes #3 = { nounwind }

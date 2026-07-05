; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all

	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.JHUFF_TBL = type { [17 x i8], [256 x i8], i32 }
	%struct.JQUANT_TBL = type { [64 x i16], i32 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.anon = type { [8 x i32], [48 x i8] }
	%struct.backing_store_info = type { ptr, ptr, ptr, ptr, [64 x i8] }
	%struct.jpeg_color_deconverter = type { ptr, ptr }
	%struct.jpeg_color_quantizer = type { ptr, ptr, ptr, ptr }
	%struct.jpeg_common_struct = type { ptr, ptr, ptr, i32, i32 }
	%struct.jpeg_component_info = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr }
	%struct.jpeg_d_coef_controller = type { ptr, ptr, ptr, ptr, ptr }
	%struct.jpeg_d_main_controller = type { ptr, ptr }
	%struct.jpeg_d_post_controller = type { ptr, ptr }
	%struct.jpeg_decomp_master = type { ptr, ptr, i32 }
	%struct.jpeg_decompress_struct = type { ptr, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, ptr, [4 x ptr], [4 x ptr], [4 x ptr], i32, ptr, i32, i32, [16 x i8], [16 x i8], [16 x i8], i32, i32, i8, i16, i16, i32, i8, i32, i32, i32, i32, i32, ptr, i32, [4 x ptr], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.jpeg_entropy_decoder = type { ptr, ptr }
	%struct.jpeg_error_mgr = type { ptr, ptr, ptr, ptr, ptr, i32, %struct.anon, i32, i32, ptr, i32, ptr, i32, i32 }
	%struct.jpeg_input_controller = type { ptr, ptr, ptr, ptr, i32, i32 }
	%struct.jpeg_inverse_dct = type { ptr, [10 x ptr] }
	%struct.jpeg_marker_reader = type { ptr, ptr, ptr, ptr, [16 x ptr], i32, i32, i32, i32 }
	%struct.jpeg_memory_mgr = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.jpeg_progress_mgr = type { ptr, i32, i32, i32, i32 }
	%struct.jpeg_source_mgr = type { ptr, i32, ptr, ptr, ptr, ptr, ptr }
	%struct.jpeg_upsampler = type { ptr, ptr, i32 }
	%struct.jvirt_barray_control = type { ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, %struct.backing_store_info }
	%struct.jvirt_sarray_control = type { ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, %struct.backing_store_info }

define void @jpeg_idct_float(ptr nocapture %cinfo, ptr nocapture %compptr, ptr nocapture %coef_block, ptr nocapture %output_buf, i32 %output_col) nounwind {
entry:
	%workspace = alloca [64 x float], align 4		; <ptr> [#uses=11]
	%0 = load ptr, ptr undef, align 4		; <ptr> [#uses=5]
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=11]
	%tmp39 = add i32 %indvar, 8		; <i32> [#uses=0]
	%tmp41 = add i32 %indvar, 16		; <i32> [#uses=2]
	%scevgep42 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp41		; <ptr> [#uses=1]
	%tmp43 = add i32 %indvar, 24		; <i32> [#uses=1]
	%scevgep44 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp43		; <ptr> [#uses=1]
	%tmp45 = add i32 %indvar, 32		; <i32> [#uses=1]
	%scevgep46 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp45		; <ptr> [#uses=1]
	%tmp47 = add i32 %indvar, 40		; <i32> [#uses=1]
	%scevgep48 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp47		; <ptr> [#uses=1]
	%tmp49 = add i32 %indvar, 48		; <i32> [#uses=1]
	%scevgep50 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp49		; <ptr> [#uses=1]
	%tmp51 = add i32 %indvar, 56		; <i32> [#uses=1]
	%scevgep52 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp51		; <ptr> [#uses=1]
	%wsptr.119 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %indvar		; <ptr> [#uses=1]
	%tmp54 = shl i32 %indvar, 2		; <i32> [#uses=1]
	%scevgep76 = getelementptr i8, ptr undef, i32 %tmp54		; <ptr> [#uses=1]
	%scevgep79 = getelementptr i16, ptr %coef_block, i32 %tmp41		; <ptr> [#uses=0]
	%inptr.117 = getelementptr i16, ptr %coef_block, i32 %indvar		; <ptr> [#uses=1]
	%1 = load i16, ptr null, align 2		; <i16> [#uses=1]
	%2 = load i16, ptr undef, align 2		; <i16> [#uses=1]
	%3 = load i16, ptr %inptr.117, align 2		; <i16> [#uses=1]
	%4 = sitofp i16 %3 to float		; <float> [#uses=1]
	%5 = load float, ptr %scevgep76, align 4		; <float> [#uses=1]
	%6 = fmul float %4, %5		; <float> [#uses=1]
	%7 = fsub float %6, undef		; <float> [#uses=2]
	%8 = fmul float undef, 0x3FF6A09E60000000		; <float> [#uses=1]
	%9 = fsub float %8, 0.000000e+00		; <float> [#uses=2]
	%10 = fadd float undef, 0.000000e+00		; <float> [#uses=2]
	%11 = fadd float %7, %9		; <float> [#uses=2]
	%12 = fsub float %7, %9		; <float> [#uses=2]
	%13 = sitofp i16 %1 to float		; <float> [#uses=1]
	%14 = fmul float %13, undef		; <float> [#uses=2]
	%15 = sitofp i16 %2 to float		; <float> [#uses=1]
	%16 = load float, ptr undef, align 4		; <float> [#uses=1]
	%17 = fmul float %15, %16		; <float> [#uses=1]
	%18 = fadd float %14, undef		; <float> [#uses=2]
	%19 = fsub float %14, undef		; <float> [#uses=2]
	%20 = fadd float undef, %17		; <float> [#uses=2]
	%21 = fadd float %20, %18		; <float> [#uses=3]
	%22 = fsub float %20, %18		; <float> [#uses=1]
	%23 = fmul float %22, 0x3FF6A09E60000000		; <float> [#uses=1]
	%24 = fadd float %19, undef		; <float> [#uses=1]
	%25 = fmul float %24, 0x3FFD906BC0000000		; <float> [#uses=2]
	%26 = fmul float undef, 0x3FF1517A80000000		; <float> [#uses=1]
	%27 = fsub float %26, %25		; <float> [#uses=1]
	%28 = fmul float %19, 0xC004E7AEA0000000		; <float> [#uses=1]
	%29 = fadd float %28, %25		; <float> [#uses=1]
	%30 = fsub float %29, %21		; <float> [#uses=3]
	%31 = fsub float %23, %30		; <float> [#uses=3]
	%32 = fadd float %27, %31		; <float> [#uses=1]
	%33 = fadd float %10, %21		; <float> [#uses=1]
	store float %33, ptr %wsptr.119, align 4
	%34 = fsub float %10, %21		; <float> [#uses=1]
	store float %34, ptr %scevgep52, align 4
	%35 = fadd float %11, %30		; <float> [#uses=1]
	store float %35, ptr null, align 4
	%36 = fsub float %11, %30		; <float> [#uses=1]
	store float %36, ptr %scevgep50, align 4
	%37 = fadd float %12, %31		; <float> [#uses=1]
	store float %37, ptr %scevgep42, align 4
	%38 = fsub float %12, %31		; <float> [#uses=1]
	store float %38, ptr %scevgep48, align 4
	%39 = fadd float undef, %32		; <float> [#uses=1]
	store float %39, ptr %scevgep46, align 4
	store float undef, ptr %scevgep44, align 4
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 undef, label %bb6, label %bb

bb6:		; preds = %bb
	%.sum10 = add i32 %output_col, 1		; <i32> [#uses=1]
	%.sum8 = add i32 %output_col, 6		; <i32> [#uses=1]
	%.sum6 = add i32 %output_col, 2		; <i32> [#uses=1]
	%.sum = add i32 %output_col, 3		; <i32> [#uses=1]
	br label %bb8

bb8:		; preds = %bb8, %bb6
	%ctr.116 = phi i32 [ 0, %bb6 ], [ %88, %bb8 ]		; <i32> [#uses=3]
	%scevgep = getelementptr ptr, ptr %output_buf, i32 %ctr.116		; <ptr> [#uses=1]
	%tmp = shl i32 %ctr.116, 3		; <i32> [#uses=5]
	%tmp2392 = or i32 %tmp, 4		; <i32> [#uses=1]
	%scevgep24 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp2392		; <ptr> [#uses=1]
	%tmp2591 = or i32 %tmp, 2		; <i32> [#uses=1]
	%scevgep26 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp2591		; <ptr> [#uses=1]
	%tmp2790 = or i32 %tmp, 6		; <i32> [#uses=1]
	%scevgep28 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp2790		; <ptr> [#uses=1]
	%tmp3586 = or i32 %tmp, 7		; <i32> [#uses=0]
	%wsptr.215 = getelementptr [64 x float], ptr %workspace, i32 0, i32 %tmp		; <ptr> [#uses=1]
	%40 = load ptr, ptr %scevgep, align 4		; <ptr> [#uses=4]
	%41 = load float, ptr %wsptr.215, align 4		; <float> [#uses=1]
	%42 = load float, ptr %scevgep24, align 4		; <float> [#uses=1]
	%43 = fadd float %41, %42		; <float> [#uses=1]
	%44 = load float, ptr %scevgep26, align 4		; <float> [#uses=1]
	%45 = load float, ptr %scevgep28, align 4		; <float> [#uses=1]
	%46 = fadd float %44, %45		; <float> [#uses=1]
	%47 = fsub float %43, %46		; <float> [#uses=2]
	%48 = fsub float undef, 0.000000e+00		; <float> [#uses=1]
	%49 = fadd float 0.000000e+00, undef		; <float> [#uses=1]
	%50 = fptosi float %49 to i32		; <i32> [#uses=1]
	%51 = add i32 %50, 4		; <i32> [#uses=1]
	%52 = lshr i32 %51, 3		; <i32> [#uses=1]
	%53 = and i32 %52, 1023		; <i32> [#uses=1]
	%.sum14 = add i32 %53, 128		; <i32> [#uses=1]
	%54 = getelementptr i8, ptr %0, i32 %.sum14		; <ptr> [#uses=1]
	%55 = load i8, ptr %54, align 1		; <i8> [#uses=1]
	store i8 %55, ptr null, align 1
	%56 = getelementptr i8, ptr %40, i32 %.sum10		; <ptr> [#uses=1]
	store i8 0, ptr %56, align 1
	%57 = load i8, ptr null, align 1		; <i8> [#uses=1]
	%58 = getelementptr i8, ptr %40, i32 %.sum8		; <ptr> [#uses=1]
	store i8 %57, ptr %58, align 1
	%59 = fadd float undef, %48		; <float> [#uses=1]
	%60 = fptosi float %59 to i32		; <i32> [#uses=1]
	%61 = add i32 %60, 4		; <i32> [#uses=1]
	%62 = lshr i32 %61, 3		; <i32> [#uses=1]
	%63 = and i32 %62, 1023		; <i32> [#uses=1]
	%.sum7 = add i32 %63, 128		; <i32> [#uses=1]
	%64 = getelementptr i8, ptr %0, i32 %.sum7		; <ptr> [#uses=1]
	%65 = load i8, ptr %64, align 1		; <i8> [#uses=1]
	%66 = getelementptr i8, ptr %40, i32 %.sum6		; <ptr> [#uses=1]
	store i8 %65, ptr %66, align 1
	%67 = fptosi float undef to i32		; <i32> [#uses=1]
	%68 = add i32 %67, 4		; <i32> [#uses=1]
	%69 = lshr i32 %68, 3		; <i32> [#uses=1]
	%70 = and i32 %69, 1023		; <i32> [#uses=1]
	%.sum5 = add i32 %70, 128		; <i32> [#uses=1]
	%71 = getelementptr i8, ptr %0, i32 %.sum5		; <ptr> [#uses=1]
	%72 = load i8, ptr %71, align 1		; <i8> [#uses=1]
	store i8 %72, ptr undef, align 1
	%73 = fadd float %47, undef		; <float> [#uses=1]
	%74 = fptosi float %73 to i32		; <i32> [#uses=1]
	%75 = add i32 %74, 4		; <i32> [#uses=1]
	%76 = lshr i32 %75, 3		; <i32> [#uses=1]
	%77 = and i32 %76, 1023		; <i32> [#uses=1]
	%.sum3 = add i32 %77, 128		; <i32> [#uses=1]
	%78 = getelementptr i8, ptr %0, i32 %.sum3		; <ptr> [#uses=1]
	%79 = load i8, ptr %78, align 1		; <i8> [#uses=1]
	store i8 %79, ptr undef, align 1
	%80 = fsub float %47, undef		; <float> [#uses=1]
	%81 = fptosi float %80 to i32		; <i32> [#uses=1]
	%82 = add i32 %81, 4		; <i32> [#uses=1]
	%83 = lshr i32 %82, 3		; <i32> [#uses=1]
	%84 = and i32 %83, 1023		; <i32> [#uses=1]
	%.sum1 = add i32 %84, 128		; <i32> [#uses=1]
	%85 = getelementptr i8, ptr %0, i32 %.sum1		; <ptr> [#uses=1]
	%86 = load i8, ptr %85, align 1		; <i8> [#uses=1]
	%87 = getelementptr i8, ptr %40, i32 %.sum		; <ptr> [#uses=1]
	store i8 %86, ptr %87, align 1
	%88 = add i32 %ctr.116, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %88, 8		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb8

return:		; preds = %bb8
	ret void
}

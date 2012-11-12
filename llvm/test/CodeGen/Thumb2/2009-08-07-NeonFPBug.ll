; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.JHUFF_TBL = type { [17 x i8], [256 x i8], i32 }
	%struct.JQUANT_TBL = type { [64 x i16], i32 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.anon = type { [8 x i32], [48 x i8] }
	%struct.backing_store_info = type { void (%struct.jpeg_common_struct*, %struct.backing_store_info*, i8*, i32, i32)*, void (%struct.jpeg_common_struct*, %struct.backing_store_info*, i8*, i32, i32)*, void (%struct.jpeg_common_struct*, %struct.backing_store_info*)*, %struct.FILE*, [64 x i8] }
	%struct.jpeg_color_deconverter = type { void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*, i8***, i32, i8**, i32)* }
	%struct.jpeg_color_quantizer = type { void (%struct.jpeg_decompress_struct*, i32)*, void (%struct.jpeg_decompress_struct*, i8**, i8**, i32)*, void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)* }
	%struct.jpeg_common_struct = type { %struct.jpeg_error_mgr*, %struct.jpeg_memory_mgr*, %struct.jpeg_progress_mgr*, i32, i32 }
	%struct.jpeg_component_info = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.JQUANT_TBL*, i8* }
	%struct.jpeg_d_coef_controller = type { void (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*, i8***)*, %struct.jvirt_barray_control** }
	%struct.jpeg_d_main_controller = type { void (%struct.jpeg_decompress_struct*, i32)*, void (%struct.jpeg_decompress_struct*, i8**, i32*, i32)* }
	%struct.jpeg_d_post_controller = type { void (%struct.jpeg_decompress_struct*, i32)*, void (%struct.jpeg_decompress_struct*, i8***, i32*, i32, i8**, i32*, i32)* }
	%struct.jpeg_decomp_master = type { void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)*, i32 }
	%struct.jpeg_decompress_struct = type { %struct.jpeg_error_mgr*, %struct.jpeg_memory_mgr*, %struct.jpeg_progress_mgr*, i32, i32, %struct.jpeg_source_mgr*, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8**, i32, i32, i32, i32, i32, [64 x i32]*, [4 x %struct.JQUANT_TBL*], [4 x %struct.JHUFF_TBL*], [4 x %struct.JHUFF_TBL*], i32, %struct.jpeg_component_info*, i32, i32, [16 x i8], [16 x i8], [16 x i8], i32, i32, i8, i16, i16, i32, i8, i32, i32, i32, i32, i32, i8*, i32, [4 x %struct.jpeg_component_info*], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, %struct.jpeg_decomp_master*, %struct.jpeg_d_main_controller*, %struct.jpeg_d_coef_controller*, %struct.jpeg_d_post_controller*, %struct.jpeg_input_controller*, %struct.jpeg_marker_reader*, %struct.jpeg_entropy_decoder*, %struct.jpeg_inverse_dct*, %struct.jpeg_upsampler*, %struct.jpeg_color_deconverter*, %struct.jpeg_color_quantizer* }
	%struct.jpeg_entropy_decoder = type { void (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*, [64 x i16]**)* }
	%struct.jpeg_error_mgr = type { void (%struct.jpeg_common_struct*)*, void (%struct.jpeg_common_struct*, i32)*, void (%struct.jpeg_common_struct*)*, void (%struct.jpeg_common_struct*, i8*)*, void (%struct.jpeg_common_struct*)*, i32, %struct.anon, i32, i32, i8**, i32, i8**, i32, i32 }
	%struct.jpeg_input_controller = type { i32 (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*)*, i32, i32 }
	%struct.jpeg_inverse_dct = type { void (%struct.jpeg_decompress_struct*)*, [10 x void (%struct.jpeg_decompress_struct*, %struct.jpeg_component_info*, i16*, i8**, i32)*] }
	%struct.jpeg_marker_reader = type { void (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*)*, [16 x i32 (%struct.jpeg_decompress_struct*)*], i32, i32, i32, i32 }
	%struct.jpeg_memory_mgr = type { i8* (%struct.jpeg_common_struct*, i32, i32)*, i8* (%struct.jpeg_common_struct*, i32, i32)*, i8** (%struct.jpeg_common_struct*, i32, i32, i32)*, [64 x i16]** (%struct.jpeg_common_struct*, i32, i32, i32)*, %struct.jvirt_sarray_control* (%struct.jpeg_common_struct*, i32, i32, i32, i32, i32)*, %struct.jvirt_barray_control* (%struct.jpeg_common_struct*, i32, i32, i32, i32, i32)*, void (%struct.jpeg_common_struct*)*, i8** (%struct.jpeg_common_struct*, %struct.jvirt_sarray_control*, i32, i32, i32)*, [64 x i16]** (%struct.jpeg_common_struct*, %struct.jvirt_barray_control*, i32, i32, i32)*, void (%struct.jpeg_common_struct*, i32)*, void (%struct.jpeg_common_struct*)*, i32 }
	%struct.jpeg_progress_mgr = type { void (%struct.jpeg_common_struct*)*, i32, i32, i32, i32 }
	%struct.jpeg_source_mgr = type { i8*, i32, void (%struct.jpeg_decompress_struct*)*, i32 (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*, i32)*, i32 (%struct.jpeg_decompress_struct*, i32)*, void (%struct.jpeg_decompress_struct*)* }
	%struct.jpeg_upsampler = type { void (%struct.jpeg_decompress_struct*)*, void (%struct.jpeg_decompress_struct*, i8***, i32*, i32, i8**, i32*, i32)*, i32 }
	%struct.jvirt_barray_control = type { [64 x i16]**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.jvirt_barray_control*, %struct.backing_store_info }
	%struct.jvirt_sarray_control = type { i8**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.jvirt_sarray_control*, %struct.backing_store_info }

define void @jpeg_idct_float(%struct.jpeg_decompress_struct* nocapture %cinfo, %struct.jpeg_component_info* nocapture %compptr, i16* nocapture %coef_block, i8** nocapture %output_buf, i32 %output_col) nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%0 = load float* undef, align 4		; <float> [#uses=1]
	%1 = fmul float undef, %0		; <float> [#uses=2]
	%tmp73 = add i32 0, 224		; <i32> [#uses=1]
	%scevgep74 = getelementptr i8* null, i32 %tmp73		; <i8*> [#uses=1]
	%scevgep7475 = bitcast i8* %scevgep74 to float*		; <float*> [#uses=1]
	%2 = load float* null, align 4		; <float> [#uses=1]
	%3 = fmul float 0.000000e+00, %2		; <float> [#uses=2]
	%4 = fadd float %1, %3		; <float> [#uses=1]
	%5 = fsub float %1, %3		; <float> [#uses=2]
	%6 = fadd float undef, 0.000000e+00		; <float> [#uses=2]
	%7 = fmul float undef, 0x3FF6A09E60000000		; <float> [#uses=1]
	%8 = fsub float %7, %6		; <float> [#uses=2]
	%9 = fsub float %4, %6		; <float> [#uses=1]
	%10 = fadd float %5, %8		; <float> [#uses=2]
	%11 = fsub float %5, %8		; <float> [#uses=1]
	%12 = sitofp i16 undef to float		; <float> [#uses=1]
	%13 = fmul float %12, 0.000000e+00		; <float> [#uses=2]
	%14 = sitofp i16 undef to float		; <float> [#uses=1]
	%15 = load float* %scevgep7475, align 4		; <float> [#uses=1]
	%16 = fmul float %14, %15		; <float> [#uses=2]
	%17 = fadd float undef, undef		; <float> [#uses=2]
	%18 = fadd float %13, %16		; <float> [#uses=2]
	%19 = fsub float %13, %16		; <float> [#uses=1]
	%20 = fadd float %18, %17		; <float> [#uses=2]
	%21 = fsub float %18, %17		; <float> [#uses=1]
	%22 = fmul float %21, 0x3FF6A09E60000000		; <float> [#uses=1]
	%23 = fmul float undef, 0x3FFD906BC0000000		; <float> [#uses=2]
	%24 = fmul float %19, 0x3FF1517A80000000		; <float> [#uses=1]
	%25 = fsub float %24, %23		; <float> [#uses=1]
	%26 = fadd float undef, %23		; <float> [#uses=1]
	%27 = fsub float %26, %20		; <float> [#uses=3]
	%28 = fsub float %22, %27		; <float> [#uses=2]
	%29 = fadd float %25, %28		; <float> [#uses=1]
	%30 = fadd float undef, %20		; <float> [#uses=1]
	store float %30, float* undef, align 4
	%31 = fadd float %10, %27		; <float> [#uses=1]
	store float %31, float* undef, align 4
	%32 = fsub float %10, %27		; <float> [#uses=1]
	store float %32, float* undef, align 4
	%33 = fadd float %11, %28		; <float> [#uses=1]
	store float %33, float* undef, align 4
	%34 = fsub float %9, %29		; <float> [#uses=1]
	store float %34, float* undef, align 4
	br label %bb
}

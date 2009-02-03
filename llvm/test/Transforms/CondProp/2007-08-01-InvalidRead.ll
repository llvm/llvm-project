; RUN: llvm-as < %s | opt -inline -tailduplicate -condprop -simplifycfg -disable-output
; PR1575
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%struct.DCTtab = type { i8, i8, i8 }
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct.VLCtab = type { i8, i8 }
	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
	%struct.layer_data = type { i32, [2048 x i8], i8*, [16 x i8], i32, i8*, i32, i32, [64 x i32], [64 x i32], [64 x i32], [64 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [12 x [64 x i16]] }
@ld = external global %struct.layer_data*		; <%struct.layer_data**> [#uses=1]
@System_Stream_Flag = external global i32		; <i32*> [#uses=0]
@Fault_Flag = external global i32		; <i32*> [#uses=2]
@picture_coding_type = external global i32		; <i32*> [#uses=1]
@DCTtabnext = external global [12 x %struct.DCTtab]		; <[12 x %struct.DCTtab]*> [#uses=0]
@DCTtab0 = external global [60 x %struct.DCTtab]		; <[60 x %struct.DCTtab]*> [#uses=0]
@DCTtab1 = external global [8 x %struct.DCTtab]		; <[8 x %struct.DCTtab]*> [#uses=0]
@DCTtab2 = external global [16 x %struct.DCTtab]		; <[16 x %struct.DCTtab]*> [#uses=0]
@DCTtab3 = external global [16 x %struct.DCTtab]		; <[16 x %struct.DCTtab]*> [#uses=0]
@DCTtab4 = external global [16 x %struct.DCTtab]		; <[16 x %struct.DCTtab]*> [#uses=0]
@DCTtab5 = external global [16 x %struct.DCTtab]		; <[16 x %struct.DCTtab]*> [#uses=0]
@DCTtab6 = external global [16 x %struct.DCTtab]		; <[16 x %struct.DCTtab]*> [#uses=0]
@Quiet_Flag = external global i32		; <i32*> [#uses=0]
@.str = external constant [51 x i8]		; <[51 x i8]*> [#uses=0]
@stderr = external global %struct.FILE*		; <%struct.FILE**> [#uses=0]
@.str1 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@scan = external global [2 x [64 x i8]]		; <[2 x [64 x i8]]*> [#uses=0]
@DCTtabfirst = external global [12 x %struct.DCTtab]		; <[12 x %struct.DCTtab]*> [#uses=0]
@.str2 = external constant [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str3 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@base = external global %struct.layer_data		; <%struct.layer_data*> [#uses=1]
@enhan = external global %struct.layer_data		; <%struct.layer_data*> [#uses=0]
@chroma_format = external global i32		; <i32*> [#uses=2]
@intra_dc_precision = external global i32		; <i32*> [#uses=0]
@intra_vlc_format = external global i32		; <i32*> [#uses=0]
@DCTtab0a = external global [252 x %struct.DCTtab]		; <[252 x %struct.DCTtab]*> [#uses=0]
@DCTtab1a = external global [8 x %struct.DCTtab]		; <[8 x %struct.DCTtab]*> [#uses=0]
@.str4 = external constant [51 x i8]		; <[51 x i8]*> [#uses=0]
@.str5 = external constant [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str6 = external constant [44 x i8]		; <[44 x i8]*> [#uses=0]
@.str7 = external constant [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str8 = external constant [44 x i8]		; <[44 x i8]*> [#uses=0]
@Temporal_Reference_Base = external global i32		; <i32*> [#uses=0]
@True_Framenum_max = external global i32		; <i32*> [#uses=0]
@Temporal_Reference_GOP_Reset.b = external global i1		; <i1*> [#uses=0]
@frame_rate_Table = external constant [16 x double]		; <[16 x double]*> [#uses=0]
@.str9 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@horizontal_size = external global i32		; <i32*> [#uses=0]
@vertical_size = external global i32		; <i32*> [#uses=0]
@aspect_ratio_information = external global i32		; <i32*> [#uses=0]
@frame_rate_code = external global i32		; <i32*> [#uses=0]
@bit_rate_value = external global i32		; <i32*> [#uses=0]
@.str110 = external constant [18 x i8]		; <[18 x i8]*> [#uses=0]
@vbv_buffer_size = external global i32		; <i32*> [#uses=0]
@constrained_parameters_flag = external global i32		; <i32*> [#uses=0]
@default_intra_quantizer_matrix = external global [64 x i8]		; <[64 x i8]*> [#uses=0]
@drop_flag = external global i32		; <i32*> [#uses=0]
@hour = external global i32		; <i32*> [#uses=0]
@minute = external global i32		; <i32*> [#uses=0]
@.str211 = external constant [27 x i8]		; <[27 x i8]*> [#uses=0]
@sec = external global i32		; <i32*> [#uses=0]
@frame = external global i32		; <i32*> [#uses=0]
@closed_gop = external global i32		; <i32*> [#uses=0]
@broken_link = external global i32		; <i32*> [#uses=0]
@temporal_reference = external global i32		; <i32*> [#uses=0]
@vbv_delay = external global i32		; <i32*> [#uses=0]
@full_pel_forward_vector = external global i32		; <i32*> [#uses=0]
@forward_f_code = external global i32		; <i32*> [#uses=0]
@full_pel_backward_vector = external global i32		; <i32*> [#uses=0]
@backward_f_code = external global i32		; <i32*> [#uses=1]
@Non_Linear_quantizer_scale = external global [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str312 = external constant [37 x i8]		; <[37 x i8]*> [#uses=0]
@layer_id = external global i32		; <i32*> [#uses=0]
@profile_and_level_indication = external global i32		; <i32*> [#uses=0]
@progressive_sequence = external global i32		; <i32*> [#uses=0]
@.str413 = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@low_delay = external global i32		; <i32*> [#uses=0]
@frame_rate_extension_n = external global i32		; <i32*> [#uses=0]
@frame_rate_extension_d = external global i32		; <i32*> [#uses=0]
@frame_rate = external global double		; <double*> [#uses=0]
@profile = external global i32		; <i32*> [#uses=0]
@level = external global i32		; <i32*> [#uses=0]
@bit_rate = external global double		; <double*> [#uses=0]
@video_format = external global i32		; <i32*> [#uses=0]
@color_description = external global i32		; <i32*> [#uses=0]
@color_primaries = external global i32		; <i32*> [#uses=0]
@transfer_characteristics = external global i32		; <i32*> [#uses=0]
@matrix_coefficients = external global i32		; <i32*> [#uses=0]
@display_horizontal_size = external global i32		; <i32*> [#uses=0]
@.str514 = external constant [27 x i8]		; <[27 x i8]*> [#uses=0]
@display_vertical_size = external global i32		; <i32*> [#uses=0]
@lower_layer_prediction_horizontal_size = external global i32		; <i32*> [#uses=0]
@.str615 = external constant [30 x i8]		; <[30 x i8]*> [#uses=0]
@lower_layer_prediction_vertical_size = external global i32		; <i32*> [#uses=0]
@horizontal_subsampling_factor_m = external global i32		; <i32*> [#uses=0]
@horizontal_subsampling_factor_n = external global i32		; <i32*> [#uses=0]
@vertical_subsampling_factor_m = external global i32		; <i32*> [#uses=0]
@vertical_subsampling_factor_n = external global i32		; <i32*> [#uses=0]
@.str716 = external constant [38 x i8]		; <[38 x i8]*> [#uses=0]
@repeat_first_field = external global i32		; <i32*> [#uses=0]
@top_field_first = external global i32		; <i32*> [#uses=0]
@picture_structure = external global i32		; <i32*> [#uses=0]
@frame_center_horizontal_offset = external global [3 x i32]		; <[3 x i32]*> [#uses=0]
@.str817 = external constant [44 x i8]		; <[44 x i8]*> [#uses=0]
@frame_center_vertical_offset = external global [3 x i32]		; <[3 x i32]*> [#uses=0]
@.str918 = external constant [45 x i8]		; <[45 x i8]*> [#uses=0]
@f_code = external global [2 x [2 x i32]]		; <[2 x [2 x i32]]*> [#uses=0]
@frame_pred_frame_dct = external global i32		; <i32*> [#uses=0]
@concealment_motion_vectors = external global i32		; <i32*> [#uses=1]
@chroma_420_type = external global i32		; <i32*> [#uses=0]
@progressive_frame = external global i32		; <i32*> [#uses=0]
@composite_display_flag = external global i32		; <i32*> [#uses=0]
@v_axis = external global i32		; <i32*> [#uses=0]
@field_sequence = external global i32		; <i32*> [#uses=0]
@sub_carrier = external global i32		; <i32*> [#uses=0]
@burst_amplitude = external global i32		; <i32*> [#uses=0]
@sub_carrier_phase = external global i32		; <i32*> [#uses=0]
@lower_layer_temporal_reference = external global i32		; <i32*> [#uses=0]
@.str10 = external constant [55 x i8]		; <[55 x i8]*> [#uses=0]
@lower_layer_horizontal_offset = external global i32		; <i32*> [#uses=0]
@.str11 = external constant [56 x i8]		; <[56 x i8]*> [#uses=0]
@lower_layer_vertical_offset = external global i32		; <i32*> [#uses=0]
@spatial_temporal_weight_code_table_index = external global i32		; <i32*> [#uses=0]
@lower_layer_progressive_frame = external global i32		; <i32*> [#uses=0]
@lower_layer_deinterlaced_field_select = external global i32		; <i32*> [#uses=0]
@.str12 = external constant [36 x i8]		; <[36 x i8]*> [#uses=0]
@copyright_flag = external global i32		; <i32*> [#uses=0]
@copyright_identifier = external global i32		; <i32*> [#uses=0]
@original_or_copy = external global i32		; <i32*> [#uses=0]
@.str13 = external constant [40 x i8]		; <[40 x i8]*> [#uses=0]
@copyright_number_1 = external global i32		; <i32*> [#uses=0]
@.str14 = external constant [41 x i8]		; <[41 x i8]*> [#uses=0]
@copyright_number_2 = external global i32		; <i32*> [#uses=0]
@.str15 = external constant [40 x i8]		; <[40 x i8]*> [#uses=0]
@copyright_number_3 = external global i32		; <i32*> [#uses=0]
@Verbose_Flag = external global i32		; <i32*> [#uses=0]
@.str16 = external constant [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str17 = external constant [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str18 = external constant [27 x i8]		; <[27 x i8]*> [#uses=0]
@.str19 = external constant [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str20 = external constant [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str21 = external constant [25 x i8]		; <[25 x i8]*> [#uses=0]
@.str22 = external constant [25 x i8]		; <[25 x i8]*> [#uses=0]
@temporal_reference_old.2592 = external global i32		; <i32*> [#uses=0]
@temporal_reference_wrap.2591.b = external global i1		; <i1*> [#uses=0]
@True_Framenum = external global i32		; <i32*> [#uses=0]
@Second_Field = external global i32		; <i32*> [#uses=0]
@.str23 = external constant [29 x i8]		; <[29 x i8]*> [#uses=0]
@Ersatz_Flag = external global i32		; <i32*> [#uses=0]
@mb_width = external global i32		; <i32*> [#uses=0]
@mb_height = external global i32		; <i32*> [#uses=0]
@Two_Streams = external global i32		; <i32*> [#uses=0]
@.str124 = external constant [32 x i8]		; <[32 x i8]*> [#uses=0]
@stwc_table.2193 = external constant [3 x [4 x i8]]		; <[3 x [4 x i8]]*> [#uses=0]
@stwclass_table.2194 = external constant [9 x i8]		; <[9 x i8]*> [#uses=0]
@current_frame = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@Coded_Picture_Width = external global i32		; <i32*> [#uses=0]
@Chroma_Width = external global i32		; <i32*> [#uses=0]
@Clip = external global i8*		; <i8**> [#uses=0]
@.str225 = external constant [30 x i8]		; <[30 x i8]*> [#uses=0]
@.str326 = external constant [27 x i8]		; <[27 x i8]*> [#uses=0]
@block_count = external global i32		; <i32*> [#uses=1]
@auxframe = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@forward_reference_frame = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@backward_reference_frame = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@.str427 = external constant [34 x i8]		; <[34 x i8]*> [#uses=0]
@Newref_progressive_frame.2631 = external global i32		; <i32*> [#uses=0]
@Oldref_progressive_frame.2630 = external global i32		; <i32*> [#uses=0]
@Reference_IDCT_Flag = external global i32		; <i32*> [#uses=0]
@.str528 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str629 = external constant [29 x i8]		; <[29 x i8]*> [#uses=0]
@.str730 = external constant [38 x i8]		; <[38 x i8]*> [#uses=0]
@.str831 = external constant [32 x i8]		; <[32 x i8]*> [#uses=0]
@PMBtab0 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@PMBtab1 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@BMBtab0 = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@BMBtab1 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@spIMBtab = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@spPMBtab0 = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@spPMBtab1 = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@spBMBtab0 = external constant [14 x %struct.VLCtab]		; <[14 x %struct.VLCtab]*> [#uses=0]
@spBMBtab1 = external constant [12 x %struct.VLCtab]		; <[12 x %struct.VLCtab]*> [#uses=0]
@spBMBtab2 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@SNRMBtab = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@MVtab0 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@MVtab1 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@MVtab2 = external constant [12 x %struct.VLCtab]		; <[12 x %struct.VLCtab]*> [#uses=0]
@CBPtab0 = external constant [32 x %struct.VLCtab]		; <[32 x %struct.VLCtab]*> [#uses=0]
@CBPtab1 = external constant [64 x %struct.VLCtab]		; <[64 x %struct.VLCtab]*> [#uses=0]
@CBPtab2 = external constant [8 x %struct.VLCtab]		; <[8 x %struct.VLCtab]*> [#uses=0]
@MBAtab1 = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@MBAtab2 = external constant [104 x %struct.VLCtab]		; <[104 x %struct.VLCtab]*> [#uses=0]
@DClumtab0 = external constant [32 x %struct.VLCtab]		; <[32 x %struct.VLCtab]*> [#uses=0]
@DClumtab1 = external constant [16 x %struct.VLCtab]		; <[16 x %struct.VLCtab]*> [#uses=0]
@DCchromtab0 = external constant [32 x %struct.VLCtab]		; <[32 x %struct.VLCtab]*> [#uses=0]
@DCchromtab1 = external constant [32 x %struct.VLCtab]		; <[32 x %struct.VLCtab]*> [#uses=0]
@.str32 = external constant [56 x i8]		; <[56 x i8]*> [#uses=0]
@.str133 = external constant [29 x i8]		; <[29 x i8]*> [#uses=0]
@global_pic = external global i32		; <i32*> [#uses=0]
@global_MBA = external global i32		; <i32*> [#uses=0]
@.str1648 = external constant [45 x i8]		; <[45 x i8]*> [#uses=0]
@.str1749 = external constant [33 x i8]		; <[33 x i8]*> [#uses=0]
@.str1850 = external constant [42 x i8]		; <[42 x i8]*> [#uses=0]
@iclp = external global i16*		; <i16**> [#uses=0]
@iclip = external global [1024 x i16]		; <[1024 x i16]*> [#uses=0]
@c = external global [8 x [8 x double]]		; <[8 x [8 x double]]*> [#uses=0]
@Version = external global [28 x i8]		; <[28 x i8]*> [#uses=0]
@Author = external global [41 x i8]		; <[41 x i8]*> [#uses=0]
@Inverse_Table_6_9 = external global [8 x [4 x i32]]		; <[8 x [4 x i32]]*> [#uses=0]
@Main_Bitstream_Filename = external global i8*		; <i8**> [#uses=0]
@.str51 = external constant [36 x i8]		; <[36 x i8]*> [#uses=0]
@Error_Text = external global [256 x i8]		; <[256 x i8]*> [#uses=0]
@.str152 = external constant [52 x i8]		; <[52 x i8]*> [#uses=0]
@.str253 = external constant [33 x i8]		; <[33 x i8]*> [#uses=0]
@Enhancement_Layer_Bitstream_Filename = external global i8*		; <i8**> [#uses=0]
@.str354 = external constant [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str455 = external constant [22 x i8]		; <[22 x i8]*> [#uses=0]
@.str556 = external constant [30 x i8]		; <[30 x i8]*> [#uses=0]
@Coded_Picture_Height = external global i32		; <i32*> [#uses=0]
@Chroma_Height = external global i32		; <i32*> [#uses=0]
@Table_6_20.3737 = external constant [3 x i32]		; <[3 x i32]*> [#uses=0]
@.str657 = external constant [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str758 = external constant [41 x i8]		; <[41 x i8]*> [#uses=0]
@.str859 = external constant [26 x i8]		; <[26 x i8]*> [#uses=0]
@substitute_frame = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@.str960 = external constant [34 x i8]		; <[34 x i8]*> [#uses=0]
@llframe0 = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@.str1061 = external constant [24 x i8]		; <[24 x i8]*> [#uses=0]
@llframe1 = external global [3 x i8*]		; <[3 x i8*]*> [#uses=0]
@.str1162 = external constant [24 x i8]		; <[24 x i8]*> [#uses=0]
@lltmp = external global i16*		; <i16**> [#uses=0]
@.str1263 = external constant [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str1364 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str1465 = external constant [9 x i8]		; <[9 x i8]*> [#uses=0]
@.str1566 = external constant [1195 x i8]		; <[1195 x i8]*> [#uses=0]
@Output_Type = external global i32		; <i32*> [#uses=0]
@Main_Bitstream_Flag = external global i32		; <i32*> [#uses=0]
@.str1667 = external constant [55 x i8]		; <[55 x i8]*> [#uses=0]
@.str1768 = external constant [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str1869 = external constant [39 x i8]		; <[39 x i8]*> [#uses=0]
@Frame_Store_Flag = external global i32		; <i32*> [#uses=0]
@Big_Picture_Flag = external global i32		; <i32*> [#uses=0]
@.str1970 = external constant [49 x i8]		; <[49 x i8]*> [#uses=0]
@Spatial_Flag = external global i32		; <i32*> [#uses=0]
@.str2071 = external constant [39 x i8]		; <[39 x i8]*> [#uses=0]
@Lower_Layer_Picture_Filename = external global i8*		; <i8**> [#uses=0]
@Output_Picture_Filename = external global i8*		; <i8**> [#uses=0]
@.str2172 = external constant [1 x i8]		; <[1 x i8]*> [#uses=0]
@.str2273 = external constant [39 x i8]		; <[39 x i8]*> [#uses=0]
@.str2374 = external constant [49 x i8]		; <[49 x i8]*> [#uses=0]
@User_Data_Flag = external global i32		; <i32*> [#uses=0]
@.str24 = external constant [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str25 = external constant [39 x i8]		; <[39 x i8]*> [#uses=0]
@Substitute_Picture_Filename = external global i8*		; <i8**> [#uses=0]
@.str26 = external constant [47 x i8]		; <[47 x i8]*> [#uses=0]
@.str27 = external constant [55 x i8]		; <[55 x i8]*> [#uses=0]
@Display_Progressive_Flag = external global i32		; <i32*> [#uses=0]
@.str28 = external constant [21 x i8]		; <[21 x i8]*> [#uses=0]
@.str29 = external constant [2 x i8]		; <[2 x i8]*> [#uses=0]
@hiQdither = external global i32		; <i32*> [#uses=0]
@Trace_Flag = external global i32		; <i32*> [#uses=0]
@Verify_Flag = external global i32		; <i32*> [#uses=0]
@Stats_Flag = external global i32		; <i32*> [#uses=0]
@Decode_Layer = external global i32		; <i32*> [#uses=0]
@.str75 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@C.53.2124 = external constant [3 x [3 x i8]]		; <[3 x [3 x i8]]*> [#uses=0]
@.str76 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@C.60.2169 = external constant [3 x [3 x i8]]		; <[3 x [3 x i8]]*> [#uses=0]
@.str77 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str178 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str279 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str380 = external constant [11 x i8]		; <[11 x i8]*> [#uses=0]
@outfile = external global i32		; <i32*> [#uses=0]
@.str481 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@optr = external global i8*		; <i8**> [#uses=0]
@obfr = external global [4096 x i8]		; <[4096 x i8]*> [#uses=0]
@.str582 = external constant [35 x i8]		; <[35 x i8]*> [#uses=0]
@u422.3075 = external global i8*		; <i8**> [#uses=0]
@v422.3076 = external global i8*		; <i8**> [#uses=0]
@.str683 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str784 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@u444.3185 = external global i8*		; <i8**> [#uses=0]
@v444.3186 = external global i8*		; <i8**> [#uses=0]
@u422.3183 = external global i8*		; <i8**> [#uses=0]
@v422.3184 = external global i8*		; <i8**> [#uses=0]
@.str885 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str986 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@tga24.3181 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@.str1087 = external constant [14 x i8]		; <[14 x i8]*> [#uses=0]
@bgate.2952.b = external global i1		; <i1*> [#uses=0]
@previous_temporal_reference.2947 = external global i32		; <i32*> [#uses=0]
@previous_picture_coding_type.2951 = external global i32		; <i32*> [#uses=0]
@previous_anchor_temporal_reference.2949 = external global i32		; <i32*> [#uses=0]
@.str88 = external constant [40 x i8]		; <[40 x i8]*> [#uses=0]
@.str189 = external constant [31 x i8]		; <[31 x i8]*> [#uses=0]
@.str290 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str391 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str492 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str593 = external constant [49 x i8]		; <[49 x i8]*> [#uses=0]
@.str694 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str795 = external constant [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str896 = external constant [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str97 = external constant [18 x i8]		; <[18 x i8]*> [#uses=0]
@.str198 = external constant [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str299 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]

declare void @Initialize_Buffer()

declare void @Fill_Buffer()

declare i32 @read(...)

declare i32 @Get_Byte()

declare i32 @Get_Word()

declare i32 @Show_Bits(i32)

declare i32 @Get_Bits1()

declare void @Flush_Buffer(i32)

declare void @Next_Packet()

declare i32 @Get_Bits(i32)

declare void @Decode_MPEG1_Intra_Block(i32, i32*)

declare i32 @Get_Luma_DC_dct_diff()

declare i32 @Get_Chroma_DC_dct_diff()

declare i32 @puts(i8*)

declare i32 @fwrite(i8*, i32, i32, i8*)

declare void @Decode_MPEG1_Non_Intra_Block(i32)

declare void @Decode_MPEG2_Intra_Block(i32, i32*)

declare void @Decode_MPEG2_Non_Intra_Block(i32)

declare i32 @Get_Hdr()

declare i32 @Get_Bits32()

declare i32 @fprintf(%struct.FILE*, i8*, ...)

declare void @next_start_code()

declare fastcc void @sequence_header()

define internal fastcc void @group_of_pictures_header() {
entry:
	ret void
}

define internal fastcc void @picture_header() {
entry:
	unreachable
}

declare i32 @slice_header()

declare fastcc void @extension_and_user_data()

declare void @Flush_Buffer32()

declare fastcc void @sequence_extension()

declare fastcc void @sequence_display_extension()

declare fastcc void @quant_matrix_extension()

declare fastcc void @sequence_scalable_extension()

declare void @Error(i8*)

declare fastcc void @picture_display_extension()

declare fastcc void @picture_coding_extension()

declare fastcc void @picture_spatial_scalable_extension()

declare fastcc void @picture_temporal_scalable_extension()

declare fastcc void @extra_bit_information()

declare void @marker_bit(i8*)

declare fastcc void @user_data()

declare fastcc void @copyright_extension()

declare i32 @printf(i8*, ...)

declare fastcc void @Update_Temporal_Reference_Tacking_Data()

define void @Decode_Picture(i32 %bitstream_framenum, i32 %sequence_framenum) {
entry:
	%tmp16 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp16, label %bb43, label %bb22

bb22:		; preds = %entry
	ret void

bb43:		; preds = %entry
	call fastcc void @picture_data( )
	ret void
}

declare void @Substitute_Frame_Buffer(i32, i32)

define void @Spatial_Prediction() {
entry:
	ret void
}

define internal fastcc void @picture_data() {
entry:
	%tmp4 = icmp eq i32 0, 3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb8, label %bb

bb:		; preds = %entry
	ret void

bb8:		; preds = %entry
	%tmp11 = call fastcc i32 @slice( i32 0 )		; <i32> [#uses=0]
	ret void
}

define internal fastcc i32 @slice(i32 %MBAmax) {
entry:
	%tmp6 = icmp eq i32 0, 1		; <i1> [#uses=1]
	br i1 %tmp6, label %bb9, label %bb231

bb9:		; preds = %entry
	%tmp11 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp11, label %bb27, label %bb17

bb17:		; preds = %bb9
	ret i32 0

bb27:		; preds = %bb9
	%tmp31 = icmp slt i32 0, %MBAmax		; <i1> [#uses=1]
	br i1 %tmp31, label %bb110, label %bb231

resync:		; preds = %bb139
	ret i32 0

bb110:		; preds = %bb27
	%tmp113 = icmp slt i32 0, %MBAmax		; <i1> [#uses=1]
	br i1 %tmp113, label %bb131, label %bb119

bb119:		; preds = %bb110
	ret i32 0

bb131:		; preds = %bb110
	%tmp133 = icmp eq i32 0, 1		; <i1> [#uses=1]
	br i1 %tmp133, label %bb139, label %bb166

bb139:		; preds = %bb131
	%tmp144 = call fastcc i32 @decode_macroblock( i32* null, i32* null, i32* null, i32* null, i32* null, [2 x [2 x i32]]* null, i32* null, [2 x i32]* null, i32* null )		; <i32> [#uses=1]
	switch i32 %tmp144, label %bb166 [
		 i32 -1, label %bb231
		 i32 0, label %resync
	]

bb166:		; preds = %bb139, %bb131
	ret i32 0

bb231:		; preds = %bb139, %bb27, %entry
	ret i32 0
}

declare i32 @Get_macroblock_address_increment()

declare fastcc void @macroblock_modes(i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)

declare i32 @Get_macroblock_type()

declare fastcc void @Add_Block(i32, i32, i32, i32, i32)

declare fastcc void @Decode_SNR_Macroblock(i32*, i32*, i32, i32, i32*)

declare i32 @Get_coded_block_pattern()

declare fastcc void @Clear_Block(i32)

declare fastcc void @Sum_Block(i32)

declare fastcc void @Saturate(i16*)

declare fastcc void @Update_Picture_Buffers()

declare void @Output_Last_Frame_of_Sequence(i32)

declare void @Write_Frame(i8**, i32)

declare fastcc void @frame_reorder(i32, i32)

declare fastcc void @motion_compensation(i32, i32, i32, [2 x [2 x i32]]*, [2 x i32]*, i32*, i32, i32)

declare void @form_predictions(i32, i32, i32, i32, [2 x [2 x i32]]*, [2 x i32]*, i32*, i32)

declare void @Reference_IDCT(i16*)

declare void @Fast_IDCT(i16*)

declare fastcc void @skipped_macroblock(i32*, [2 x [2 x i32]]*, i32*, [2 x i32]*, i32*, i32*)

declare fastcc i32 @start_of_slice(i32*, i32*, i32*, [2 x [2 x i32]]*)

define internal fastcc i32 @decode_macroblock(i32* %macroblock_type, i32* %stwtype, i32* %stwclass, i32* %motion_type, i32* %dct_type, [2 x [2 x i32]]* %PMV, i32* %dc_dct_pred, [2 x i32]* %motion_vertical_field_select, i32* %dmvector) {
entry:
	%tmp3 = icmp eq i32 0, 1		; <i1> [#uses=1]
	br i1 %tmp3, label %bb, label %bb15

bb:		; preds = %entry
	%tmp7 = icmp slt i32 0, 3		; <i1> [#uses=1]
	br i1 %tmp7, label %bb13, label %bb14

bb13:		; preds = %bb
	br label %bb15

bb14:		; preds = %bb
	ret i32 0

bb15:		; preds = %bb13, %entry
	%tmp21 = load i32* @Fault_Flag, align 4		; <i32> [#uses=1]
	%tmp22 = icmp eq i32 %tmp21, 0		; <i1> [#uses=1]
	br i1 %tmp22, label %bb29, label %bb630

bb29:		; preds = %bb15
	%tmp33 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp33, label %bb91, label %bb39

bb39:		; preds = %bb29
	ret i32 0

bb91:		; preds = %bb29
	%tmp94 = and i32 0, 8		; <i32> [#uses=0]
	%tmp121 = load %struct.layer_data** @ld, align 4		; <%struct.layer_data*> [#uses=0]
	%tmp123 = load i32* null		; <i32> [#uses=1]
	%tmp124 = icmp eq i32 %tmp123, 0		; <i1> [#uses=1]
	br i1 %tmp124, label %bb146, label %bb130

bb130:		; preds = %bb91
	call void @motion_vectors( [2 x [2 x i32]]* %PMV, i32* %dmvector, [2 x i32]* %motion_vertical_field_select, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 )
	br label %bb157

bb146:		; preds = %bb91
	br label %bb157

bb157:		; preds = %bb146, %bb130
	%tmp159 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp159, label %bb166, label %bb630

bb166:		; preds = %bb157
	%tmp180 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp180, label %bb201, label %bb186

bb186:		; preds = %bb166
	br label %bb212

bb201:		; preds = %bb166
	%tmp205 = load i32* @backward_f_code, align 4		; <i32> [#uses=0]
	br label %bb212

bb212:		; preds = %bb201, %bb186
	%tmp214 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp214, label %bb221, label %bb630

bb221:		; preds = %bb212
	%tmp22422511 = and i32 0, 1		; <i32> [#uses=1]
	%toBool226 = icmp eq i32 %tmp22422511, 0		; <i1> [#uses=1]
	br i1 %toBool226, label %bb239, label %bb230

bb230:		; preds = %bb221
	ret i32 0

bb239:		; preds = %bb221
	%tmp241 = load i32* getelementptr (%struct.layer_data* @base, i32 0, i32 17), align 4		; <i32> [#uses=0]
	%tmp262 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp262, label %bb296, label %bb268

bb268:		; preds = %bb239
	%tmp270 = load i32* @chroma_format, align 4		; <i32> [#uses=1]
	%tmp271 = icmp eq i32 %tmp270, 2		; <i1> [#uses=1]
	br i1 %tmp271, label %bb277, label %bb282

bb277:		; preds = %bb268
	br label %bb312

bb282:		; preds = %bb268
	%tmp283 = load i32* @chroma_format, align 4		; <i32> [#uses=0]
	br label %bb312

bb296:		; preds = %bb239
	%tmp298 = load i32* %macroblock_type		; <i32> [#uses=1]
	%tmp2993009 = and i32 %tmp298, 1		; <i32> [#uses=1]
	%toBool301 = icmp eq i32 %tmp2993009, 0		; <i1> [#uses=1]
	br i1 %toBool301, label %bb312, label %bb305

bb305:		; preds = %bb296
	%tmp306 = load i32* @block_count, align 4		; <i32> [#uses=0]
	%tmp308 = add i32 0, -1		; <i32> [#uses=0]
	br label %bb312

bb312:		; preds = %bb305, %bb296, %bb282, %bb277
	%tmp313 = load i32* @Fault_Flag, align 4		; <i32> [#uses=1]
	%tmp314 = icmp eq i32 %tmp313, 0		; <i1> [#uses=1]
	br i1 %tmp314, label %bb398, label %bb630

bb346:		; preds = %cond_true404
	%toBool351 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%tmp359 = icmp ne i32 0, 0		; <i1> [#uses=2]
	br i1 %toBool351, label %bb372, label %bb355

bb355:		; preds = %bb346
	br i1 %tmp359, label %bb365, label %bb368

bb365:		; preds = %bb355
	br label %bb386

bb368:		; preds = %bb355
	call void @Decode_MPEG1_Intra_Block( i32 0, i32* %dc_dct_pred )
	br label %bb386

bb372:		; preds = %bb346
	br i1 %tmp359, label %bb382, label %bb384

bb382:		; preds = %bb372
	br label %bb386

bb384:		; preds = %bb372
	call void @Decode_MPEG1_Non_Intra_Block( i32 0 )
	br label %bb386

bb386:		; preds = %bb384, %bb382, %bb368, %bb365
	%tmp388 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp388, label %bb395, label %bb630

bb395:		; preds = %cond_true404, %bb386
	%tmp397 = add i32 0, 1		; <i32> [#uses=0]
	ret i32 0

bb398:		; preds = %bb312
	%tmp401 = icmp slt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp401, label %cond_true404, label %bb407

cond_true404:		; preds = %bb398
	%tmp340341514 = and i32 0, 0		; <i32> [#uses=1]
	%toBool342 = icmp eq i32 %tmp340341514, 0		; <i1> [#uses=1]
	br i1 %toBool342, label %bb395, label %bb346

bb407:		; preds = %bb398
	%tmp408 = load i32* @picture_coding_type, align 4		; <i32> [#uses=0]
	%tmp419 = load i32* %macroblock_type		; <i32> [#uses=1]
	%tmp420 = and i32 %tmp419, 1		; <i32> [#uses=1]
	%tmp421 = icmp eq i32 %tmp420, 0		; <i1> [#uses=0]
	%tmp442 = load i32* %macroblock_type		; <i32> [#uses=1]
	%tmp4434447 = and i32 %tmp442, 1		; <i32> [#uses=0]
	%tmp450 = load i32* @concealment_motion_vectors, align 4		; <i32> [#uses=0]
	%tmp572 = icmp eq i32 0, 4		; <i1> [#uses=1]
	br i1 %tmp572, label %bb578, label %bb630

bb578:		; preds = %bb407
	%tmp613 = getelementptr [2 x [2 x i32]]* %PMV, i32 1, i32 1, i32 1		; <i32*> [#uses=0]
	%tmp618 = getelementptr [2 x [2 x i32]]* %PMV, i32 1, i32 1, i32 0		; <i32*> [#uses=0]
	%tmp623 = getelementptr [2 x [2 x i32]]* %PMV, i32 0, i32 1, i32 1		; <i32*> [#uses=0]
	%tmp628 = getelementptr [2 x [2 x i32]]* %PMV, i32 0, i32 1, i32 0		; <i32*> [#uses=0]
	ret i32 1

bb630:		; preds = %bb407, %bb386, %bb312, %bb212, %bb157, %bb15
	%tmp.0 = phi i32 [ 0, %bb15 ], [ 0, %bb157 ], [ 0, %bb212 ], [ 0, %bb312 ], [ 0, %bb386 ], [ 1, %bb407 ]		; <i32> [#uses=1]
	ret i32 %tmp.0
}

declare void @motion_vectors([2 x [2 x i32]]*, i32*, [2 x i32]*, i32, i32, i32, i32, i32, i32, i32)

declare void @motion_vector(i32*, i32*, i32, i32, i32, i32, i32)

declare fastcc i32 @Get_I_macroblock_type()

declare fastcc i32 @Get_P_macroblock_type()

declare fastcc i32 @Get_B_macroblock_type()

declare fastcc void @Get_D_macroblock_type()

declare fastcc i32 @Get_I_Spatial_macroblock_type()

declare fastcc i32 @Get_P_Spatial_macroblock_type()

declare fastcc i32 @Get_B_Spatial_macroblock_type()

declare fastcc i32 @Get_SNR_macroblock_type()

declare i32 @Get_motion_code()

declare i32 @Get_dmvector()

declare fastcc void @idctrow(i16*)

declare fastcc void @idctcol(i16*)

declare void @Initialize_Fast_IDCT()

declare void @Initialize_Reference_IDCT()

declare double @cos(double)

declare double @floor(double)

declare fastcc void @decode_motion_vector(i32*, i32, i32, i32, i32)

declare void @Dual_Prime_Arithmetic([2 x i32]*, i32*, i32, i32)

declare i32 @main(i32, i8**)

declare i32 @open(i8*, i32, ...)

declare void @exit(i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare i32 @lseek(i32, i32, i32)

declare i32 @sprintf(i8*, i8*, ...)

declare i32 @close(i32)

declare fastcc void @Initialize_Decoder()

declare fastcc void @Initialize_Sequence()

declare void @Print_Bits(i32, i32, i32)

declare fastcc void @Process_Options(i32, i8**)

declare i32 @toupper(i32)

declare i32 @atoi(i8*)

declare fastcc i32 @Headers()

declare fastcc void @Decode_Bitstream()

declare fastcc void @Deinitialize_Sequence()

declare fastcc i32 @video_sequence(i32*)

declare void @Clear_Options()

declare fastcc void @form_prediction(i8**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare fastcc void @form_component_prediction(i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare fastcc void @Read_Lower_Layer_Component_Framewise(i32, i32, i32)

declare i8* @strcat(i8*, i8*)

declare %struct.FILE* @fopen(i8*, i8*)

declare i32 @_IO_getc(%struct.FILE*)

declare i32 @fclose(%struct.FILE*)

declare fastcc void @Read_Lower_Layer_Component_Fieldwise(i32, i32, i32)

declare fastcc void @Make_Spatial_Prediction_Frame(i32, i32, i8*, i8*, i16*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

declare fastcc void @Deinterlace(i8*, i8*, i32, i32, i32, i32)

declare fastcc void @Subsample_Vertical(i8*, i16*, i32, i32, i32, i32, i32, i32, i32)

declare fastcc void @Subsample_Horizontal(i16*, i8*, i32, i32, i32, i32, i32, i32, i32)

declare fastcc void @store_one(i8*, i8**, i32, i32, i32)

declare fastcc void @store_yuv(i8*, i8**, i32, i32, i32)

declare fastcc void @store_yuv1(i8*, i8*, i32, i32, i32, i32)

declare i32 @write(...)

declare fastcc void @store_sif(i8*, i8**, i32, i32, i32)

declare fastcc void @store_ppm_tga(i8*, i8**, i32, i32, i32, i32)

declare fastcc void @putbyte(i32)

declare fastcc void @putword(i32)

declare fastcc void @conv422to444(i8*, i8*)

declare fastcc void @conv420to422(i8*, i8*)

declare fastcc void @Read_Frame(i8*, i8**, i32)

declare fastcc i32 @Read_Components(i8*, i32)

declare fastcc void @Read_Component(i8*, i8*, i32, i32)

declare fastcc i32 @Extract_Components(i8*, i32)

declare i32 @fseek(%struct.FILE*, i32, i32)

declare i32 @fread(i8*, i32, i32, %struct.FILE*)

declare fastcc void @Copy_Frame(i8*, i8*, i32, i32, i32, i32)

declare i32 @Get_Long()

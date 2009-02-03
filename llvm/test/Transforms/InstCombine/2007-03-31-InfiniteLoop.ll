; RUN: llvm-as < %s | opt -instcombine -disable-output
; END.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
	%struct.ZZIP_FILE = type { %struct.zzip_dir*, i32, i32, i32, i32, i32, i32, i64, i8*, i64, %struct.z_stream, %struct.zzip_plugin_io* }
	%struct.anon = type { %struct.ZZIP_FILE*, i8* }
	%struct.internal_state = type { i32 }
	%struct.z_stream = type { i8*, i32, i32, i8*, i32, i32, i8*, %struct.internal_state*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i8*, i32, i32, i32 }
	%struct.zzip_dir = type { i32, i32, i32, %struct.anon, %struct.zzip_dir_hdr*, %struct.zzip_dir_hdr*, %struct.ZZIP_FILE*, %struct.zzip_dirent, i8*, i8*, i8**, %struct.zzip_plugin_io* }
	%struct.zzip_dir_hdr = type { i32, i32, i32, i32, i16, i16, i8, i16, [1 x i8] }
	%struct.zzip_dirent = type { i32, i32, i32, i16, i8*, i32, i32 }
	%struct.zzip_plugin_io = type { i32 (i8*, i32, ...)*, i32 (i32)*, i32 (i32, i8*, i32)*, i64 (i32, i64, i32)*, i64 (i32)*, i32 }

define %struct.ZZIP_FILE* @zzip_open_shared_io(%struct.ZZIP_FILE* %stream, i8* %filename, i32 %o_flags, i32 %o_modes, i8** %ext, %struct.zzip_plugin_io* %io) {
entry:
	%basename = alloca [1024 x i8], align 16		; <[1024 x i8]*> [#uses=5]
	%e = alloca i32, align 4		; <i32*> [#uses=4]
	icmp eq %struct.ZZIP_FILE* %stream, null		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_next22, label %cond_true

cond_true:		; preds = %entry
	%tmp3 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp4 = load %struct.zzip_dir** %tmp3		; <%struct.zzip_dir*> [#uses=1]
	icmp eq %struct.zzip_dir* %tmp4, null		; <i1>:1 [#uses=1]
	br i1 %1, label %cond_next22, label %cond_true5

cond_true5:		; preds = %cond_true
	icmp eq i8** %ext, null		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_true7, label %cond_next

cond_true7:		; preds = %cond_true5
	%tmp9 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp10 = load %struct.zzip_dir** %tmp9		; <%struct.zzip_dir*> [#uses=1]
	%tmp11 = getelementptr %struct.zzip_dir* %tmp10, i32 0, i32 10		; <i8***> [#uses=1]
	%tmp12 = load i8*** %tmp11		; <i8**> [#uses=1]
	br label %cond_next

cond_next:		; preds = %cond_true7, %cond_true5
	%ext_addr.0 = phi i8** [ %ext, %cond_true5 ], [ %tmp12, %cond_true7 ]		; <i8**> [#uses=2]
	icmp eq %struct.zzip_plugin_io* %io, null		; <i1>:3 [#uses=1]
	br i1 %3, label %cond_true14, label %cond_next22

cond_true14:		; preds = %cond_next
	%tmp16 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp17 = load %struct.zzip_dir** %tmp16		; <%struct.zzip_dir*> [#uses=1]
	%tmp18 = getelementptr %struct.zzip_dir* %tmp17, i32 0, i32 11		; <%struct.zzip_plugin_io**> [#uses=1]
	%tmp19 = load %struct.zzip_plugin_io** %tmp18		; <%struct.zzip_plugin_io*> [#uses=1]
	br label %cond_next22

cond_next22:		; preds = %cond_true14, %cond_next, %cond_true, %entry
	%io_addr.0 = phi %struct.zzip_plugin_io* [ %io, %entry ], [ %io, %cond_true ], [ %io, %cond_next ], [ %tmp19, %cond_true14 ]		; <%struct.zzip_plugin_io*> [#uses=2]
	%ext_addr.1 = phi i8** [ %ext, %entry ], [ %ext, %cond_true ], [ %ext_addr.0, %cond_next ], [ %ext_addr.0, %cond_true14 ]		; <i8**> [#uses=2]
	icmp eq %struct.zzip_plugin_io* %io_addr.0, null		; <i1>:4 [#uses=1]
	br i1 %4, label %cond_true24, label %cond_next26

cond_true24:		; preds = %cond_next22
	%tmp25 = call %struct.zzip_plugin_io* @zzip_get_default_io( )		; <%struct.zzip_plugin_io*> [#uses=1]
	br label %cond_next26

cond_next26:		; preds = %cond_true24, %cond_next22
	%io_addr.1 = phi %struct.zzip_plugin_io* [ %io_addr.0, %cond_next22 ], [ %tmp25, %cond_true24 ]		; <%struct.zzip_plugin_io*> [#uses=4]
	%tmp28 = and i32 %o_modes, 81920		; <i32> [#uses=1]
	icmp eq i32 %tmp28, 0		; <i1>:5 [#uses=1]
	br i1 %5, label %try_real, label %try_zzip

try_real:		; preds = %bb223, %cond_next26
	%fd160.2 = phi i32 [ undef, %cond_next26 ], [ %fd160.0, %bb223 ]		; <i32> [#uses=1]
	%len.2 = phi i32 [ undef, %cond_next26 ], [ %len.0, %bb223 ]		; <i32> [#uses=1]
	%o_flags_addr.1 = phi i32 [ %o_flags, %cond_next26 ], [ %o_flags_addr.0, %bb223 ]		; <i32> [#uses=2]
	%tmp33348 = and i32 %o_modes, 262144		; <i32> [#uses=1]
	icmp eq i32 %tmp33348, 0		; <i1>:6 [#uses=1]
	br i1 %6, label %cond_next38, label %cond_true35

cond_true35:		; preds = %try_real
	%tmp36 = call %struct.zzip_plugin_io* @zzip_get_default_io( )		; <%struct.zzip_plugin_io*> [#uses=1]
	br label %cond_next38

cond_next38:		; preds = %cond_true35, %try_real
	%iftmp.21.0 = phi %struct.zzip_plugin_io* [ %tmp36, %cond_true35 ], [ %io_addr.1, %try_real ]		; <%struct.zzip_plugin_io*> [#uses=3]
	%tmp41 = getelementptr %struct.zzip_plugin_io* %iftmp.21.0, i32 0, i32 0		; <i32 (i8*, i32, ...)**> [#uses=1]
	%tmp42 = load i32 (i8*, i32, ...)** %tmp41		; <i32 (i8*, i32, ...)*> [#uses=1]
	%tmp45 = call i32 (i8*, i32, ...)* %tmp42( i8* %filename, i32 %o_flags_addr.1 )		; <i32> [#uses=3]
	icmp eq i32 %tmp45, -1		; <i1>:7 [#uses=1]
	br i1 %7, label %cond_next67, label %cond_true47

cond_true47:		; preds = %cond_next38
	%tmp48 = call i8* @cli_calloc( i32 1, i32 108 )		; <i8*> [#uses=2]
	%tmp4849 = bitcast i8* %tmp48 to %struct.ZZIP_FILE*		; <%struct.ZZIP_FILE*> [#uses=3]
	icmp eq i8* %tmp48, null		; <i1>:8 [#uses=1]
	br i1 %8, label %cond_true51, label %cond_next58

cond_true51:		; preds = %cond_true47
	%tmp53 = getelementptr %struct.zzip_plugin_io* %iftmp.21.0, i32 0, i32 1		; <i32 (i32)**> [#uses=1]
	%tmp54 = load i32 (i32)** %tmp53		; <i32 (i32)*> [#uses=1]
	%tmp56 = call i32 %tmp54( i32 %tmp45 )		; <i32> [#uses=0]
	ret %struct.ZZIP_FILE* null

cond_next58:		; preds = %cond_true47
	%tmp60 = getelementptr %struct.ZZIP_FILE* %tmp4849, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %tmp45, i32* %tmp60
	%tmp63 = getelementptr %struct.ZZIP_FILE* %tmp4849, i32 0, i32 11		; <%struct.zzip_plugin_io**> [#uses=1]
	store %struct.zzip_plugin_io* %iftmp.21.0, %struct.zzip_plugin_io** %tmp63
	ret %struct.ZZIP_FILE* %tmp4849

cond_next67:		; preds = %cond_next38
	%tmp70716 = and i32 %o_modes, 16384		; <i32> [#uses=1]
	icmp eq i32 %tmp70716, 0		; <i1>:9 [#uses=1]
	br i1 %9, label %try_zzip, label %return

try_zzip:		; preds = %cond_next67, %cond_next26
	%fd160.3 = phi i32 [ %fd160.2, %cond_next67 ], [ undef, %cond_next26 ]		; <i32> [#uses=6]
	%len.3 = phi i32 [ %len.2, %cond_next67 ], [ undef, %cond_next26 ]		; <i32> [#uses=3]
	%o_flags_addr.3 = phi i32 [ %o_flags_addr.1, %cond_next67 ], [ %o_flags, %cond_next26 ]		; <i32> [#uses=4]
	%tmp76 = and i32 %o_flags_addr.3, 513		; <i32> [#uses=1]
	icmp eq i32 %tmp76, 0		; <i1>:10 [#uses=1]
	br i1 %10, label %cond_next80, label %cond_true77

cond_true77:		; preds = %try_zzip
	%tmp78 = call i32* @__error( )		; <i32*> [#uses=1]
	store i32 22, i32* %tmp78
	ret %struct.ZZIP_FILE* null

cond_next80:		; preds = %try_zzip
	%tmp83844 = and i32 %o_flags_addr.3, 2		; <i32> [#uses=1]
	icmp eq i32 %tmp83844, 0		; <i1>:11 [#uses=1]
	%tmp87 = xor i32 %o_flags_addr.3, 2		; <i32> [#uses=1]
	%o_flags_addr.0 = select i1 %11, i32 %o_flags_addr.3, i32 %tmp87		; <i32> [#uses=2]
	%basename90 = getelementptr [1024 x i8]* %basename, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp92 = call i8* @strcpy( i8* %basename90, i8* %filename )		; <i8*> [#uses=0]
	icmp eq %struct.ZZIP_FILE* %stream, null		; <i1>:12 [#uses=1]
	br i1 %12, label %bb219, label %cond_true94

cond_true94:		; preds = %cond_next80
	%tmp96 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp97 = load %struct.zzip_dir** %tmp96		; <%struct.zzip_dir*> [#uses=1]
	icmp eq %struct.zzip_dir* %tmp97, null		; <i1>:13 [#uses=1]
	br i1 %13, label %bb219, label %cond_true98

cond_true98:		; preds = %cond_true94
	%tmp100 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp101 = load %struct.zzip_dir** %tmp100		; <%struct.zzip_dir*> [#uses=1]
	%tmp102 = getelementptr %struct.zzip_dir* %tmp101, i32 0, i32 9		; <i8**> [#uses=1]
	%tmp103 = load i8** %tmp102		; <i8*> [#uses=1]
	icmp eq i8* %tmp103, null		; <i1>:14 [#uses=1]
	br i1 %14, label %bb219, label %cond_true104

cond_true104:		; preds = %cond_true98
	%tmp106 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp107 = load %struct.zzip_dir** %tmp106		; <%struct.zzip_dir*> [#uses=1]
	%tmp108 = getelementptr %struct.zzip_dir* %tmp107, i32 0, i32 9		; <i8**> [#uses=1]
	%tmp109 = load i8** %tmp108		; <i8*> [#uses=1]
	%tmp110 = call i32 @strlen( i8* %tmp109 )		; <i32> [#uses=7]
	%tmp112 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp113 = load %struct.zzip_dir** %tmp112		; <%struct.zzip_dir*> [#uses=1]
	%tmp114 = getelementptr %struct.zzip_dir* %tmp113, i32 0, i32 9		; <i8**> [#uses=1]
	%tmp115 = load i8** %tmp114		; <i8*> [#uses=1]
	%tmp118 = call i32 @memcmp( i8* %filename, i8* %tmp115, i32 %tmp110 )		; <i32> [#uses=1]
	icmp eq i32 %tmp118, 0		; <i1>:15 [#uses=1]
	br i1 %15, label %cond_true119, label %bb219

cond_true119:		; preds = %cond_true104
	%tmp122 = getelementptr i8* %filename, i32 %tmp110		; <i8*> [#uses=1]
	%tmp123 = load i8* %tmp122		; <i8> [#uses=1]
	icmp eq i8 %tmp123, 47		; <i1>:16 [#uses=1]
	br i1 %16, label %cond_true124, label %bb219

cond_true124:		; preds = %cond_true119
	%tmp126 = add i32 %tmp110, 1		; <i32> [#uses=1]
	%tmp128 = getelementptr i8* %filename, i32 %tmp126		; <i8*> [#uses=1]
	%tmp129 = load i8* %tmp128		; <i8> [#uses=1]
	icmp eq i8 %tmp129, 0		; <i1>:17 [#uses=1]
	br i1 %17, label %bb219, label %cond_true130

cond_true130:		; preds = %cond_true124
	%tmp134.sum = add i32 %tmp110, 1		; <i32> [#uses=1]
	%tmp135 = getelementptr i8* %filename, i32 %tmp134.sum		; <i8*> [#uses=1]
	%tmp137 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp138 = load %struct.zzip_dir** %tmp137		; <%struct.zzip_dir*> [#uses=1]
	%tmp140 = call %struct.ZZIP_FILE* @zzip_file_open( %struct.zzip_dir* %tmp138, i8* %tmp135, i32 %o_modes, i32 -1 )		; <%struct.ZZIP_FILE*> [#uses=3]
	icmp eq %struct.ZZIP_FILE* %tmp140, null		; <i1>:18 [#uses=1]
	br i1 %18, label %cond_true142, label %return

cond_true142:		; preds = %cond_true130
	%tmp144 = getelementptr %struct.ZZIP_FILE* %stream, i32 0, i32 0		; <%struct.zzip_dir**> [#uses=1]
	%tmp145 = load %struct.zzip_dir** %tmp144		; <%struct.zzip_dir*> [#uses=1]
	%tmp146 = getelementptr %struct.zzip_dir* %tmp145, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp147 = load i32* %tmp146		; <i32> [#uses=1]
	%tmp148 = call i32 @zzip_errno( i32 %tmp147 )		; <i32> [#uses=1]
	%tmp149 = call i32* @__error( )		; <i32*> [#uses=1]
	store i32 %tmp148, i32* %tmp149
	ret %struct.ZZIP_FILE* %tmp140

bb:		; preds = %bb219
	store i32 0, i32* %e
	store i8 0, i8* %tmp221
	%basename162 = getelementptr [1024 x i8]* %basename, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp166 = call i32 @__zzip_try_open( i8* %basename162, i32 %o_flags_addr.0, i8** %ext_addr.1, %struct.zzip_plugin_io* %io_addr.1 )		; <i32> [#uses=4]
	icmp eq i32 %tmp166, -1		; <i1>:19 [#uses=1]
	br i1 %19, label %bb219, label %cond_next169

cond_next169:		; preds = %bb
	%tmp173 = call %struct.zzip_dir* @zzip_dir_fdopen_ext_io( i32 %tmp166, i32* %e, i8** %ext_addr.1, %struct.zzip_plugin_io* %io_addr.1 )		; <%struct.zzip_dir*> [#uses=7]
	%tmp174 = load i32* %e		; <i32> [#uses=1]
	icmp eq i32 %tmp174, 0		; <i1>:20 [#uses=1]
	br i1 %20, label %cond_next185, label %cond_true175

cond_true175:		; preds = %cond_next169
	%tmp176 = load i32* %e		; <i32> [#uses=1]
	%tmp177 = call i32 @zzip_errno( i32 %tmp176 )		; <i32> [#uses=1]
	%tmp178 = call i32* @__error( )		; <i32*> [#uses=1]
	store i32 %tmp177, i32* %tmp178
	%tmp180 = getelementptr %struct.zzip_plugin_io* %io_addr.1, i32 0, i32 1		; <i32 (i32)**> [#uses=1]
	%tmp181 = load i32 (i32)** %tmp180		; <i32 (i32)*> [#uses=1]
	%tmp183 = call i32 %tmp181( i32 %tmp166 )		; <i32> [#uses=0]
	ret %struct.ZZIP_FILE* null

cond_next185:		; preds = %cond_next169
	%tmp186187 = ptrtoint i8* %tmp221 to i32		; <i32> [#uses=1]
	%basename188189 = ptrtoint [1024 x i8]* %basename to i32		; <i32> [#uses=1]
	%tmp190 = sub i32 %tmp186187, %basename188189		; <i32> [#uses=1]
	%tmp192.sum = add i32 %tmp190, 1		; <i32> [#uses=1]
	%tmp193 = getelementptr i8* %filename, i32 %tmp192.sum		; <i8*> [#uses=1]
	%tmp196 = call %struct.ZZIP_FILE* @zzip_file_open( %struct.zzip_dir* %tmp173, i8* %tmp193, i32 %o_modes, i32 -1 )		; <%struct.ZZIP_FILE*> [#uses=4]
	icmp eq %struct.ZZIP_FILE* %tmp196, null		; <i1>:21 [#uses=1]
	br i1 %21, label %cond_true198, label %cond_false204

cond_true198:		; preds = %cond_next185
	%tmp200 = getelementptr %struct.zzip_dir* %tmp173, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp201 = load i32* %tmp200		; <i32> [#uses=1]
	%tmp202 = call i32 @zzip_errno( i32 %tmp201 )		; <i32> [#uses=1]
	%tmp203 = call i32* @__error( )		; <i32*> [#uses=1]
	store i32 %tmp202, i32* %tmp203
	%tmp2169 = call i32 @zzip_dir_close( %struct.zzip_dir* %tmp173 )		; <i32> [#uses=0]
	ret %struct.ZZIP_FILE* %tmp196

cond_false204:		; preds = %cond_next185
	%tmp206 = getelementptr %struct.zzip_dir* %tmp173, i32 0, i32 9		; <i8**> [#uses=1]
	%tmp207 = load i8** %tmp206		; <i8*> [#uses=1]
	icmp eq i8* %tmp207, null		; <i1>:22 [#uses=1]
	br i1 %22, label %cond_true208, label %cond_next214

cond_true208:		; preds = %cond_false204
	%basename209 = getelementptr [1024 x i8]* %basename, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp210 = call i8* @strdup( i8* %basename209 )		; <i8*> [#uses=1]
	%tmp212 = getelementptr %struct.zzip_dir* %tmp173, i32 0, i32 9		; <i8**> [#uses=1]
	store i8* %tmp210, i8** %tmp212
	%tmp21610 = call i32 @zzip_dir_close( %struct.zzip_dir* %tmp173 )		; <i32> [#uses=0]
	ret %struct.ZZIP_FILE* %tmp196

cond_next214:		; preds = %cond_false204
	%tmp216 = call i32 @zzip_dir_close( %struct.zzip_dir* %tmp173 )		; <i32> [#uses=0]
	ret %struct.ZZIP_FILE* %tmp196

bb219:		; preds = %bb, %cond_true124, %cond_true119, %cond_true104, %cond_true98, %cond_true94, %cond_next80
	%fd160.0 = phi i32 [ %fd160.3, %cond_next80 ], [ %tmp166, %bb ], [ %fd160.3, %cond_true94 ], [ %fd160.3, %cond_true98 ], [ %fd160.3, %cond_true104 ], [ %fd160.3, %cond_true119 ], [ %fd160.3, %cond_true124 ]		; <i32> [#uses=1]
	%len.0 = phi i32 [ %len.3, %cond_next80 ], [ %len.0, %bb ], [ %len.3, %cond_true94 ], [ %len.3, %cond_true98 ], [ %tmp110, %cond_true104 ], [ %tmp110, %cond_true119 ], [ %tmp110, %cond_true124 ]		; <i32> [#uses=2]
	%basename220 = getelementptr [1024 x i8]* %basename, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp221 = call i8* @strrchr( i8* %basename220, i32 47 )		; <i8*> [#uses=3]
	icmp eq i8* %tmp221, null		; <i1>:23 [#uses=1]
	br i1 %23, label %bb223, label %bb

bb223:		; preds = %bb219
	%tmp2262272 = and i32 %o_modes, 16384		; <i32> [#uses=1]
	icmp eq i32 %tmp2262272, 0		; <i1>:24 [#uses=1]
	br i1 %24, label %cond_next229, label %try_real

cond_next229:		; preds = %bb223
	%tmp230 = call i32* @__error( )		; <i32*> [#uses=1]
	store i32 2, i32* %tmp230
	ret %struct.ZZIP_FILE* null

return:		; preds = %cond_true130, %cond_next67
	%retval.0 = phi %struct.ZZIP_FILE* [ null, %cond_next67 ], [ %tmp140, %cond_true130 ]		; <%struct.ZZIP_FILE*> [#uses=1]
	ret %struct.ZZIP_FILE* %retval.0
}

declare i32 @zzip_dir_close(%struct.zzip_dir*)

declare i8* @strrchr(i8*, i32)

declare %struct.ZZIP_FILE* @zzip_file_open(%struct.zzip_dir*, i8*, i32, i32)

declare i8* @cli_calloc(i32, i32)

declare i32 @zzip_errno(i32)

declare i32* @__error()

declare %struct.zzip_plugin_io* @zzip_get_default_io()

declare i8* @strcpy(i8*, i8*)

declare i32 @strlen(i8*)

declare i32 @memcmp(i8*, i8*, i32)

declare i32 @__zzip_try_open(i8*, i32, i8**, %struct.zzip_plugin_io*)

declare %struct.zzip_dir* @zzip_dir_fdopen_ext_io(i32, i32*, i8**, %struct.zzip_plugin_io*)

declare i8* @strdup(i8*)

; RUN: opt < %s -loop-reduce -S | grep "phi\>" | count 8
; PR2570

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@g_14 = internal global i32 1		; <i32*> [#uses=1]
@g_39 = internal global i16 -5		; <i16*> [#uses=2]
@g_43 = internal global i32 -6		; <i32*> [#uses=3]
@g_33 = internal global i32 -1269044541		; <i32*> [#uses=1]
@g_137 = internal global i32 8		; <i32*> [#uses=1]
@g_82 = internal global i32 -5		; <i32*> [#uses=3]
@g_91 = internal global i32 1		; <i32*> [#uses=1]
@g_197 = internal global i32 1		; <i32*> [#uses=4]
@g_207 = internal global i32 1		; <i32*> [#uses=2]
@g_222 = internal global i16 4165		; <i16*> [#uses=1]
@g_247 = internal global i8 -21		; <i8*> [#uses=2]
@g_260 = internal global i32 1		; <i32*> [#uses=2]
@g_221 = internal global i16 -17503		; <i16*> [#uses=3]
@g_267 = internal global i16 1		; <i16*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32 (i32, i32, i16, i32, i8, i32)* @func_44 to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @func_44(i32 %p_45, i32 %p_46, i16 zeroext  %p_48, i32 %p_49, i8 zeroext  %p_50, i32 %p_52) nounwind  {
entry:
	tail call i32 @func_116( i8 zeroext  2 ) nounwind 		; <i32>:0 [#uses=0]
	tail call i32 @func_63( i16 signext  2 ) nounwind 		; <i32>:1 [#uses=1]
	load i16* @g_39, align 2		; <i16>:2 [#uses=1]
	tail call i32 @func_63( i16 signext  %2 ) nounwind 		; <i32>:3 [#uses=1]
	trunc i32 %3 to i16		; <i16>:4 [#uses=1]
	and i16 %4, 1		; <i16>:5 [#uses=1]
	trunc i32 %p_52 to i8		; <i8>:6 [#uses=1]
	trunc i32 %1 to i16		; <i16>:7 [#uses=1]
	tail call i32 @func_74( i16 zeroext  %5, i8 zeroext  %6, i16 zeroext  %7, i16 zeroext  0 ) nounwind 		; <i32>:8 [#uses=0]
	tail call i32 @func_124( i32 544824386 ) nounwind 		; <i32>:9 [#uses=0]
	zext i8 %p_50 to i32		; <i32>:10 [#uses=1]
	load i32* @g_43, align 4		; <i32>:11 [#uses=1]
	icmp sle i32 %10, %11		; <i1>:12 [#uses=1]
	zext i1 %12 to i32		; <i32>:13 [#uses=2]
	load i8* @g_247, align 1		; <i8>:14 [#uses=1]
	trunc i32 %p_45 to i16		; <i16>:15 [#uses=1]
	zext i8 %14 to i16		; <i16>:16 [#uses=1]
	tail call i32 @func_74( i16 zeroext  %15, i8 zeroext  0, i16 zeroext  %16, i16 zeroext  23618 ) nounwind 		; <i32>:17 [#uses=4]
	icmp slt i32 %17, 0		; <i1>:18 [#uses=1]
	br i1 %18, label %bb162, label %bb152

bb152:		; preds = %entry
	lshr i32 2147483647, %13		; <i32>:19 [#uses=1]
	icmp slt i32 %19, %17		; <i1>:20 [#uses=1]
	select i1 %20, i32 0, i32 %13		; <i32>:21 [#uses=1]
	%.348 = shl i32 %17, %21		; <i32> [#uses=1]
	br label %bb162

bb162:		; preds = %bb152, %entry
	%.0346 = phi i32 [ %.348, %bb152 ], [ %17, %entry ]		; <i32> [#uses=1]
	tail call i32 @func_124( i32 1 ) nounwind 		; <i32>:22 [#uses=1]
	mul i32 %22, %.0346		; <i32>:23 [#uses=1]
	icmp slt i32 %p_45, 0		; <i1>:24 [#uses=1]
	icmp ugt i32 %p_45, 31		; <i1>:25 [#uses=1]
	%or.cond = or i1 %24, %25		; <i1> [#uses=1]
	br i1 %or.cond, label %bb172, label %bb168

bb168:		; preds = %bb162
	lshr i32 2147483647, %p_45		; <i32>:26 [#uses=1]
	shl i32 1392859848, %p_45		; <i32>:27 [#uses=1]
	icmp slt i32 %26, 1392859848		; <i1>:28 [#uses=1]
	%.op355 = add i32 %27, 38978		; <i32> [#uses=1]
	%phitmp = select i1 %28, i32 1392898826, i32 %.op355		; <i32> [#uses=1]
	br label %bb172

bb172:		; preds = %bb168, %bb162
	%.0343 = phi i32 [ %phitmp, %bb168 ], [ 1392898826, %bb162 ]		; <i32> [#uses=2]
	tail call i32 @func_84( i32 1, i16 zeroext  0, i16 zeroext  8 ) nounwind 		; <i32>:29 [#uses=0]
	icmp eq i32 %.0343, 0		; <i1>:30 [#uses=1]
	%.0341 = select i1 %30, i32 1, i32 %.0343		; <i32> [#uses=1]
	urem i32 %23, %.0341		; <i32>:31 [#uses=1]
	load i32* @g_137, align 4		; <i32>:32 [#uses=4]
	icmp slt i32 %32, 0		; <i1>:33 [#uses=1]
	br i1 %33, label %bb202, label %bb198

bb198:		; preds = %bb172
	%not. = icmp slt i32 %32, 1073741824		; <i1> [#uses=1]
	zext i1 %not. to i32		; <i32>:34 [#uses=1]
	%.351 = shl i32 %32, %34		; <i32> [#uses=1]
	br label %bb202

bb202:		; preds = %bb198, %bb172
	%.0335 = phi i32 [ %.351, %bb198 ], [ %32, %bb172 ]		; <i32> [#uses=1]
	icmp ne i32 %31, %.0335		; <i1>:35 [#uses=1]
	zext i1 %35 to i32		; <i32>:36 [#uses=1]
	tail call i32 @func_128( i32 %36 ) nounwind 		; <i32>:37 [#uses=0]
	icmp eq i32 %p_45, 293685862		; <i1>:38 [#uses=1]
	br i1 %38, label %bb205, label %bb311

bb205:		; preds = %bb202
	icmp sgt i32 %p_46, 214		; <i1>:39 [#uses=1]
	zext i1 %39 to i32		; <i32>:40 [#uses=2]
	tail call i32 @func_128( i32 %40 ) nounwind 		; <i32>:41 [#uses=0]
	icmp sgt i32 %p_46, 65532		; <i1>:42 [#uses=1]
	zext i1 %42 to i16		; <i16>:43 [#uses=1]
	tail call i32 @func_74( i16 zeroext  23618, i8 zeroext  -29, i16 zeroext  %43, i16 zeroext  1 ) nounwind 		; <i32>:44 [#uses=2]
	tail call i32 @func_103( i16 zeroext  -869 ) nounwind 		; <i32>:45 [#uses=0]
	udiv i32 %44, 34162		; <i32>:46 [#uses=1]
	icmp ult i32 %44, 34162		; <i1>:47 [#uses=1]
	%.0331 = select i1 %47, i32 1, i32 %46		; <i32> [#uses=1]
	urem i32 293685862, %.0331		; <i32>:48 [#uses=1]
	tail call i32 @func_112( i32 %p_52, i16 zeroext  1 ) nounwind 		; <i32>:49 [#uses=0]
	icmp eq i32 %p_52, 0		; <i1>:50 [#uses=2]
	br i1 %50, label %bb222, label %bb215

bb215:		; preds = %bb205
	zext i16 %p_48 to i32		; <i32>:51 [#uses=1]
	icmp eq i16 %p_48, 0		; <i1>:52 [#uses=1]
	%.0329 = select i1 %52, i32 1, i32 %51		; <i32> [#uses=1]
	udiv i32 -1, %.0329		; <i32>:53 [#uses=1]
	icmp eq i32 %53, 0		; <i1>:54 [#uses=1]
	br i1 %54, label %bb222, label %bb223

bb222:		; preds = %bb215, %bb205
	br label %bb223

bb223:		; preds = %bb222, %bb215
	%iftmp.437.0 = phi i32 [ 0, %bb222 ], [ 1, %bb215 ]		; <i32> [#uses=1]
	load i32* @g_91, align 4		; <i32>:55 [#uses=3]
	tail call i32 @func_103( i16 zeroext  4 ) nounwind 		; <i32>:56 [#uses=0]
	tail call i32 @func_112( i32 0, i16 zeroext  -31374 ) nounwind 		; <i32>:57 [#uses=0]
	load i32* @g_197, align 4		; <i32>:58 [#uses=1]
	tail call i32 @func_124( i32 28156 ) nounwind 		; <i32>:59 [#uses=1]
	load i32* @g_260, align 4		; <i32>:60 [#uses=1]
	load i32* @g_43, align 4		; <i32>:61 [#uses=1]
	xor i32 %61, %60		; <i32>:62 [#uses=1]
	mul i32 %62, %59		; <i32>:63 [#uses=1]
	trunc i32 %63 to i8		; <i8>:64 [#uses=1]
	trunc i32 %58 to i16		; <i16>:65 [#uses=1]
	tail call i32 @func_74( i16 zeroext  0, i8 zeroext  %64, i16 zeroext  %65, i16 zeroext  0 ) nounwind 		; <i32>:66 [#uses=2]
	icmp slt i32 %66, 0		; <i1>:67 [#uses=1]
	icmp slt i32 %55, 0		; <i1>:68 [#uses=1]
	icmp ugt i32 %55, 31		; <i1>:69 [#uses=1]
	or i1 %68, %69		; <i1>:70 [#uses=1]
	%or.cond352 = or i1 %70, %67		; <i1> [#uses=1]
	select i1 %or.cond352, i32 0, i32 %55		; <i32>:71 [#uses=1]
	%.353 = ashr i32 %66, %71		; <i32> [#uses=2]
	load i16* @g_221, align 2		; <i16>:72 [#uses=1]
	zext i16 %72 to i32		; <i32>:73 [#uses=1]
	icmp ugt i32 %.353, 31		; <i1>:74 [#uses=1]
	select i1 %74, i32 0, i32 %.353		; <i32>:75 [#uses=1]
	%.0323 = lshr i32 %73, %75		; <i32> [#uses=1]
	add i32 %.0323, %iftmp.437.0		; <i32>:76 [#uses=1]
	and i32 %48, 255		; <i32>:77 [#uses=2]
	add i32 %77, 2042556439		; <i32>:78 [#uses=1]
	load i32* @g_207, align 4		; <i32>:79 [#uses=2]
	icmp ugt i32 %79, 31		; <i1>:80 [#uses=1]
	select i1 %80, i32 0, i32 %79		; <i32>:81 [#uses=1]
	%.0320 = lshr i32 %77, %81		; <i32> [#uses=1]
	icmp ne i32 %78, %.0320		; <i1>:82 [#uses=1]
	zext i1 %82 to i8		; <i8>:83 [#uses=1]
	tail call i32 @func_25( i8 zeroext  %83 ) nounwind 		; <i32>:84 [#uses=1]
	xor i32 %84, 1		; <i32>:85 [#uses=1]
	load i32* @g_197, align 4		; <i32>:86 [#uses=1]
	add i32 %86, 1		; <i32>:87 [#uses=1]
	add i32 %87, %85		; <i32>:88 [#uses=1]
	icmp ugt i32 %76, %88		; <i1>:89 [#uses=1]
	br i1 %89, label %bb241, label %bb311

bb241:		; preds = %bb223
	store i16 -9, i16* @g_221, align 2
	udiv i32 %p_52, 1538244727		; <i32>:90 [#uses=1]
	load i32* @g_207, align 4		; <i32>:91 [#uses=1]
	sub i32 %91, %90		; <i32>:92 [#uses=1]
	load i32* @g_14, align 4		; <i32>:93 [#uses=1]
	trunc i32 %93 to i16		; <i16>:94 [#uses=1]
	trunc i32 %p_46 to i16		; <i16>:95 [#uses=2]
	sub i16 %94, %95		; <i16>:96 [#uses=1]
	load i32* @g_197, align 4		; <i32>:97 [#uses=1]
	trunc i32 %97 to i16		; <i16>:98 [#uses=1]
	tail call i32 @func_55( i32 -346178830, i16 zeroext  %98, i16 zeroext  %95 ) nounwind 		; <i32>:99 [#uses=0]
	zext i16 %p_48 to i32		; <i32>:100 [#uses=1]
	load i8* @g_247, align 1		; <i8>:101 [#uses=1]
	zext i8 %101 to i32		; <i32>:102 [#uses=1]
	sub i32 %100, %102		; <i32>:103 [#uses=1]
	tail call i32 @func_55( i32 %103, i16 zeroext  -2972, i16 zeroext  %96 ) nounwind 		; <i32>:104 [#uses=0]
	xor i32 %92, 2968		; <i32>:105 [#uses=1]
	load i32* @g_197, align 4		; <i32>:106 [#uses=1]
	icmp ugt i32 %105, %106		; <i1>:107 [#uses=1]
	zext i1 %107 to i32		; <i32>:108 [#uses=1]
	store i32 %108, i32* @g_33, align 4
	br label %bb248

bb248:		; preds = %bb284, %bb241
	%p_49_addr.1.reg2mem.0 = phi i32 [ 0, %bb241 ], [ %134, %bb284 ]		; <i32> [#uses=2]
	%p_48_addr.2.reg2mem.0 = phi i16 [ %p_48, %bb241 ], [ %p_48_addr.1, %bb284 ]		; <i16> [#uses=1]
	%p_46_addr.1.reg2mem.0 = phi i32 [ %p_46, %bb241 ], [ %133, %bb284 ]		; <i32> [#uses=1]
	%p_45_addr.1.reg2mem.0 = phi i32 [ %p_45, %bb241 ], [ %p_45_addr.0, %bb284 ]		; <i32> [#uses=2]
	tail call i32 @func_63( i16 signext  1 ) nounwind 		; <i32>:109 [#uses=1]
	icmp eq i32 %109, 0		; <i1>:110 [#uses=1]
	br i1 %110, label %bb272.thread, label %bb255.thread

bb272.thread:		; preds = %bb248
	store i32 1, i32* @g_82
	load i16* @g_267, align 2		; <i16>:111 [#uses=1]
	icmp eq i16 %111, 0		; <i1>:112 [#uses=1]
	br i1 %112, label %bb311.loopexit.split, label %bb268

bb255.thread:		; preds = %bb248
	load i32* @g_260, align 4		; <i32>:113 [#uses=1]
	sub i32 %113, %p_52		; <i32>:114 [#uses=1]
	and i32 %114, -20753		; <i32>:115 [#uses=1]
	icmp ne i32 %115, 0		; <i1>:116 [#uses=1]
	zext i1 %116 to i16		; <i16>:117 [#uses=1]
	store i16 %117, i16* @g_221, align 2
	br label %bb284

bb268:		; preds = %bb268, %bb272.thread
	%indvar = phi i32 [ 0, %bb272.thread ], [ %g_82.tmp.0, %bb268 ]		; <i32> [#uses=2]
	%p_46_addr.0.reg2mem.0 = phi i32 [ %p_46_addr.1.reg2mem.0, %bb272.thread ], [ %119, %bb268 ]		; <i32> [#uses=1]
	%g_82.tmp.0 = add i32 %indvar, 1		; <i32> [#uses=2]
	trunc i32 %p_46_addr.0.reg2mem.0 to i16		; <i16>:118 [#uses=1]
	and i32 %g_82.tmp.0, 28156		; <i32>:119 [#uses=1]
	add i32 %indvar, 2		; <i32>:120 [#uses=4]
	icmp sgt i32 %120, -1		; <i1>:121 [#uses=1]
	br i1 %121, label %bb268, label %bb274.split

bb274.split:		; preds = %bb268
	store i32 %120, i32* @g_82
	br i1 %50, label %bb279, label %bb276

bb276:		; preds = %bb274.split
	store i16 0, i16* @g_222, align 2
	br label %bb284

bb279:		; preds = %bb274.split
	icmp eq i32 %120, 0		; <i1>:122 [#uses=1]
	%.0317 = select i1 %122, i32 1, i32 %120		; <i32> [#uses=1]
	udiv i32 -8, %.0317		; <i32>:123 [#uses=1]
	trunc i32 %123 to i16		; <i16>:124 [#uses=1]
	br label %bb284

bb284:		; preds = %bb279, %bb276, %bb255.thread
	%p_49_addr.0 = phi i32 [ %p_49_addr.1.reg2mem.0, %bb279 ], [ %p_49_addr.1.reg2mem.0, %bb276 ], [ 0, %bb255.thread ]		; <i32> [#uses=1]
	%p_48_addr.1 = phi i16 [ %124, %bb279 ], [ %118, %bb276 ], [ %p_48_addr.2.reg2mem.0, %bb255.thread ]		; <i16> [#uses=1]
	%p_45_addr.0 = phi i32 [ %p_45_addr.1.reg2mem.0, %bb279 ], [ %p_45_addr.1.reg2mem.0, %bb276 ], [ 8, %bb255.thread ]		; <i32> [#uses=3]
	load i32* @g_43, align 4		; <i32>:125 [#uses=1]
	trunc i32 %125 to i8		; <i8>:126 [#uses=1]
	tail call i32 @func_116( i8 zeroext  %126 ) nounwind 		; <i32>:127 [#uses=0]
	lshr i32 65255, %p_45_addr.0		; <i32>:128 [#uses=1]
	icmp ugt i32 %p_45_addr.0, 31		; <i1>:129 [#uses=1]
	%.op = lshr i32 %128, 31		; <i32> [#uses=1]
	%.op.op = xor i32 %.op, 1		; <i32> [#uses=1]
	%.354..lobit.not = select i1 %129, i32 1, i32 %.op.op		; <i32> [#uses=1]
	load i16* @g_39, align 2		; <i16>:130 [#uses=1]
	zext i16 %130 to i32		; <i32>:131 [#uses=1]
	icmp slt i32 %.354..lobit.not, %131		; <i1>:132 [#uses=1]
	zext i1 %132 to i32		; <i32>:133 [#uses=1]
	add i32 %p_49_addr.0, 1		; <i32>:134 [#uses=2]
	icmp sgt i32 %134, -1		; <i1>:135 [#uses=1]
	br i1 %135, label %bb248, label %bb307

bb307:		; preds = %bb284
	tail call i32 @func_103( i16 zeroext  0 ) nounwind 		; <i32>:136 [#uses=0]
	ret i32 %40

bb311.loopexit.split:		; preds = %bb272.thread
	store i32 1, i32* @g_82
	ret i32 1

bb311:		; preds = %bb223, %bb202
	%.0 = phi i32 [ 1, %bb202 ], [ 0, %bb223 ]		; <i32> [#uses=1]
	ret i32 %.0
}

declare i32 @func_25(i8 zeroext ) nounwind 

declare i32 @func_55(i32, i16 zeroext , i16 zeroext ) nounwind 

declare i32 @func_63(i16 signext ) nounwind 

declare i32 @func_74(i16 zeroext , i8 zeroext , i16 zeroext , i16 zeroext ) nounwind 

declare i32 @func_84(i32, i16 zeroext , i16 zeroext ) nounwind 

declare i32 @func_103(i16 zeroext ) nounwind 

declare i32 @func_124(i32) nounwind 

declare i32 @func_128(i32) nounwind 

declare i32 @func_116(i8 zeroext ) nounwind 

declare i32 @func_112(i32, i16 zeroext ) nounwind 

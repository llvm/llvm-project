; RUN: llc < %s -mcpu=yonah
; PR1427

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%struct.DrawHelper = type { void (i32, %struct.QT_FT_Span*, i8*)*, void (i32, %struct.QT_FT_Span*, i8*)*, void (%struct.QRasterBuffer*, i32, i32, i32, i8*, i32, i32, i32)*, void (%struct.QRasterBuffer*, i32, i32, i32, i8*, i32, i32, i32)*, void (%struct.QRasterBuffer*, i32, i32, i32, i32, i32)* }
	%struct.QBasicAtomic = type { i32 }
	%struct.QClipData = type { i32, %"struct.QClipData::ClipLine"*, i32, i32, %struct.QT_FT_Span*, i32, i32, i32, i32 }
	%"struct.QClipData::ClipLine" = type { i32, %struct.QT_FT_Span* }
	%struct.QRasterBuffer = type { %struct.QRect, %struct.QRegion, %struct.QClipData*, %struct.QClipData*, i8, i32, i32, %struct.DrawHelper*, i32, i32, i32, i8* }
	%struct.QRect = type { i32, i32, i32, i32 }
	%struct.QRegion = type { %"struct.QRegion::QRegionData"* }
	%"struct.QRegion::QRegionData" = type { %struct.QBasicAtomic, %struct._XRegion*, i8*, %struct.QRegionPrivate* }
	%struct.QRegionPrivate = type opaque
	%struct.QT_FT_Span = type { i16, i16, i16, i8 }
	%struct._XRegion = type opaque

define void @_Z19qt_bitmapblit16_sseP13QRasterBufferiijPKhiii(%struct.QRasterBuffer* %rasterBuffer, i32 %x, i32 %y, i32 %color, i8* %src, i32 %width, i32 %height, i32 %stride) {
entry:
	%tmp528 = bitcast <8 x i8> zeroinitializer to <2 x i32>		; <<2 x i32>> [#uses=1]
	%tmp529 = and <2 x i32> %tmp528, bitcast (<4 x i16> < i16 -32640, i16 16448, i16 8224, i16 4112 > to <2 x i32>)		; <<2 x i32>> [#uses=1]
	%tmp542 = bitcast <2 x i32> %tmp529 to <4 x i16>		; <<4 x i16>> [#uses=1]
	%tmp543 = add <4 x i16> %tmp542, < i16 0, i16 16448, i16 24672, i16 28784 >		; <<4 x i16>> [#uses=1]
	%tmp555 = bitcast <4 x i16> %tmp543 to <8 x i8>		; <<8 x i8>> [#uses=1]
        %tmp556 = bitcast <8 x i8> %tmp555 to x86_mmx
        %tmp557 = bitcast <8 x i8> zeroinitializer to x86_mmx
	tail call void @llvm.x86.mmx.maskmovq( x86_mmx %tmp557, x86_mmx %tmp556, i8* null )
	ret void
}

declare void @llvm.x86.mmx.maskmovq(x86_mmx, x86_mmx, i8*)

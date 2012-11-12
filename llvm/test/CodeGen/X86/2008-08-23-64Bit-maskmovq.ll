; RUN: llc < %s -march=x86-64

	%struct.DrawHelper = type { void (i32, %struct.QT_FT_Span*, i8*)*, void (i32, %struct.QT_FT_Span*, i8*)*, void (%struct.QRasterBuffer*, i32, i32, i32, i8*, i32, i32, i32)*, void (%struct.QRasterBuffer*, i32, i32, i32, i8*, i32, i32, i32)*, void (%struct.QRasterBuffer*, i32, i32, i32, i32, i32)* }
	%struct.QBasicAtomic = type { i32 }
	%struct.QClipData = type { i32, %"struct.QClipData::ClipLine"*, i32, i32, %struct.QT_FT_Span*, i32, i32, i32, i32 }
	%"struct.QClipData::ClipLine" = type { i32, %struct.QT_FT_Span* }
	%struct.QRasterBuffer = type { %struct.QRect, %struct.QRect, %struct.QRegion, %struct.QRegion, %struct.QClipData*, %struct.QClipData*, i8, i8, i32, i32, i32, i32, %struct.DrawHelper*, i32, i32, i32, i8* }
	%struct.QRect = type { i32, i32, i32, i32 }
	%struct.QRegion = type { %"struct.QRegion::QRegionData"* }
	%"struct.QRegion::QRegionData" = type { %struct.QBasicAtomic, %struct._XRegion*, i8*, %struct.QRegionPrivate* }
	%struct.QRegionPrivate = type opaque
	%struct.QT_FT_Span = type { i16, i16, i16, i8 }
	%struct._XRegion = type opaque

define hidden void @_Z24qt_bitmapblit16_sse3dnowP13QRasterBufferiijPKhiii(%struct.QRasterBuffer* %rasterBuffer, i32 %x, i32 %y, i32 %color, i8* %src, i32 %width, i32 %height, i32 %stride) nounwind {
entry:
	br i1 false, label %bb.nph144.split, label %bb133

bb.nph144.split:		; preds = %entry
        %tmp = bitcast <8 x i8> zeroinitializer to x86_mmx
        %tmp2 = bitcast <8 x i8> zeroinitializer to x86_mmx
	tail call void @llvm.x86.mmx.maskmovq( x86_mmx %tmp, x86_mmx %tmp2, i8* null ) nounwind
	unreachable

bb133:		; preds = %entry
	ret void
}

declare void @llvm.x86.mmx.maskmovq(x86_mmx, x86_mmx, i8*) nounwind

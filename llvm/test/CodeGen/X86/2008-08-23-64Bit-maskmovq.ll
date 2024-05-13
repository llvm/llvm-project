; RUN: llc < %s -mtriple=x86_64--

	%struct.DrawHelper = type { ptr, ptr, ptr, ptr, ptr }
	%struct.QBasicAtomic = type { i32 }
	%struct.QClipData = type { i32, ptr, i32, i32, ptr, i32, i32, i32, i32 }
	%"struct.QClipData::ClipLine" = type { i32, ptr }
	%struct.QRasterBuffer = type { %struct.QRect, %struct.QRect, %struct.QRegion, %struct.QRegion, ptr, ptr, i8, i8, i32, i32, i32, i32, ptr, i32, i32, i32, ptr }
	%struct.QRect = type { i32, i32, i32, i32 }
	%struct.QRegion = type { ptr }
	%"struct.QRegion::QRegionData" = type { %struct.QBasicAtomic, ptr, ptr, ptr }
	%struct.QRegionPrivate = type opaque
	%struct.QT_FT_Span = type { i16, i16, i16, i8 }
	%struct._XRegion = type opaque

define hidden void @_Z24qt_bitmapblit16_sse3dnowP13QRasterBufferiijPKhiii(ptr %rasterBuffer, i32 %x, i32 %y, i32 %color, ptr %src, i32 %width, i32 %height, i32 %stride) nounwind {
entry:
	br i1 false, label %bb.nph144.split, label %bb133

bb.nph144.split:		; preds = %entry
        %tmp = bitcast <8 x i8> zeroinitializer to x86_mmx
        %tmp2 = bitcast <8 x i8> zeroinitializer to x86_mmx
	tail call void @llvm.x86.mmx.maskmovq( x86_mmx %tmp, x86_mmx %tmp2, ptr null ) nounwind
	unreachable

bb133:		; preds = %entry
	ret void
}

declare void @llvm.x86.mmx.maskmovq(x86_mmx, x86_mmx, ptr) nounwind

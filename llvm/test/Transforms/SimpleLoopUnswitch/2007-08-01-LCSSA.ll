; RUN: opt < %s -passes='loop(simple-loop-unswitch),instcombine' -verify-memoryssa -disable-output
	%struct.ClassDef = type { %struct.QByteArray, %struct.QByteArray, %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", i8, i8, %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", %"struct.QList<ArgumentDef>", %"struct.QMap<QByteArray,QByteArray>", %"struct.QList<ArgumentDef>", %"struct.QMap<QByteArray,QByteArray>", i32, i32 }
	%struct.FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i32, i32, [40 x i8] }
	%struct.Generator = type { ptr, ptr, %"struct.QList<ArgumentDef>", %struct.QByteArray, %"struct.QList<ArgumentDef>" }
	%struct.QBasicAtomic = type { i32 }
	%struct.QByteArray = type { ptr }
	%"struct.QByteArray::Data" = type { %struct.QBasicAtomic, i32, i32, ptr, [1 x i8] }
	%"struct.QList<ArgumentDef>" = type { %"struct.QList<ArgumentDef>::._19" }
	%"struct.QList<ArgumentDef>::._19" = type { %struct.QListData }
	%struct.QListData = type { ptr }
	%"struct.QListData::Data" = type { %struct.QBasicAtomic, i32, i32, i32, i8, [1 x ptr] }
	%"struct.QMap<QByteArray,QByteArray>" = type { %"struct.QMap<QByteArray,QByteArray>::._56" }
	%"struct.QMap<QByteArray,QByteArray>::._56" = type { ptr }
	%struct.QMapData = type { ptr, [12 x ptr], %struct.QBasicAtomic, i32, i32, i32, i8 }
	%struct._IO_marker = type { ptr, ptr, i32 }
@.str9 = external constant [1 x i8]		; <ptr> [#uses=1]

declare i32 @strcmp(ptr, ptr)

define i32 @_ZN9Generator6strregEPKc(ptr %this, ptr %s) {
entry:
	%s_addr.0 = select i1 false, ptr @.str9, ptr %s		; <ptr> [#uses=2]
	%tmp122 = icmp eq ptr %s_addr.0, null		; <i1> [#uses=1]
	br label %bb184

bb55:		; preds = %bb184
	ret i32 0

bb88:		; preds = %bb184
	br i1 %tmp122, label %bb154, label %bb128

bb128:		; preds = %bb88
	%tmp138 = call i32 @strcmp( ptr null, ptr %s_addr.0 )		; <i32> [#uses=1]
	%iftmp.37.0.in4 = icmp eq i32 %tmp138, 0		; <i1> [#uses=1]
	br i1 %iftmp.37.0.in4, label %bb250, label %bb166

bb154:		; preds = %bb88
	br i1 false, label %bb250, label %bb166

bb166:		; preds = %bb154, %bb128
	%tmp175 = add i32 %idx.0, 1		; <i32> [#uses=1]
	%tmp177 = add i32 %tmp175, 0		; <i32> [#uses=1]
	%tmp181 = add i32 %tmp177, 0		; <i32> [#uses=1]
	%tmp183 = add i32 %i33.0, 1		; <i32> [#uses=1]
	br label %bb184

bb184:		; preds = %bb166, %entry
	%i33.0 = phi i32 [ 0, %entry ], [ %tmp183, %bb166 ]		; <i32> [#uses=2]
	%idx.0 = phi i32 [ 0, %entry ], [ %tmp181, %bb166 ]		; <i32> [#uses=2]
	%tmp49 = icmp slt i32 %i33.0, 0		; <i1> [#uses=1]
	br i1 %tmp49, label %bb88, label %bb55

bb250:		; preds = %bb154, %bb128
	ret i32 %idx.0
}

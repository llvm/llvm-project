; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; END.
	%struct..4._102 = type { %struct.QVectorData* }
	%struct..5._125 = type { %struct.QMapData* }
	%struct.QAbstractTextDocumentLayout = type { %struct.QObject }
	%struct.QBasicAtomic = type { i32 }
	%struct.QFont = type { %struct.QFontPrivate*, i32 }
	%struct.QFontMetrics = type { %struct.QFontPrivate* }
	%struct.QFontPrivate = type opaque
	%"struct.QFragmentMap<QTextBlockData>" = type { %struct.QFragmentMapData }
	%struct.QFragmentMapData = type { %"struct.QFragmentMapData::._154", i32 }
	%"struct.QFragmentMapData::._154" = type { %"struct.QFragmentMapData::Header"* }
	%"struct.QFragmentMapData::Header" = type { i32, i32, i32, i32, i32, i32, i32, i32 }
	%"struct.QHash<uint,QHashDummyValue>" = type { %"struct.QHash<uint,QHashDummyValue>::._152" }
	%"struct.QHash<uint,QHashDummyValue>::._152" = type { %struct.QHashData* }
	%struct.QHashData = type { %"struct.QHashData::Node"*, %"struct.QHashData::Node"**, %struct.QBasicAtomic, i32, i32, i16, i16, i32, i8 }
	%"struct.QHashData::Node" = type { %"struct.QHashData::Node"*, i32 }
	%"struct.QList<QObject*>::._92" = type { %struct.QListData }
	%"struct.QList<QPointer<QObject> >" = type { %"struct.QList<QObject*>::._92" }
	%struct.QListData = type { %"struct.QListData::Data"* }
	%"struct.QListData::Data" = type { %struct.QBasicAtomic, i32, i32, i32, i8, [1 x i8*] }
	%"struct.QMap<QUrl,QVariant>" = type { %struct..5._125 }
	%struct.QMapData = type { %"struct.QMapData::Node"*, [12 x %"struct.QMapData::Node"*], %struct.QBasicAtomic, i32, i32, i32, i8 }
	%"struct.QMapData::Node" = type { %"struct.QMapData::Node"*, [1 x %"struct.QMapData::Node"*] }
	%struct.QObject = type { i32 (...)**, %struct.QObjectData* }
	%struct.QObjectData = type { i32 (...)**, %struct.QObject*, %struct.QObject*, %"struct.QList<QPointer<QObject> >", i8, [3 x i8], i32, i32 }
	%struct.QObjectPrivate = type { %struct.QObjectData, i32, %struct.QObject*, %"struct.QList<QPointer<QObject> >", %"struct.QVector<QAbstractTextDocumentLayout::Selection>", %struct.QString }
	%struct.QPaintDevice = type { i32 (...)**, i16 }
	%struct.QPainter = type { %struct.QPainterPrivate* }
	%struct.QPainterPrivate = type opaque
	%struct.QPointF = type { double, double }
	%struct.QPrinter = type { %struct.QPaintDevice, %struct.QPrinterPrivate* }
	%struct.QPrinterPrivate = type opaque
	%struct.QRectF = type { double, double, double, double }
	%"struct.QSet<uint>" = type { %"struct.QHash<uint,QHashDummyValue>" }
	%"struct.QSharedDataPointer<QTextFormatPrivate>" = type { %struct.QTextFormatPrivate* }
	%struct.QString = type { %"struct.QString::Data"* }
	%"struct.QString::Data" = type { %struct.QBasicAtomic, i32, i32, i16*, i8, i8, [1 x i16] }
	%struct.QTextBlockFormat = type { %struct.QTextFormat }
	%struct.QTextBlockGroup = type { %struct.QAbstractTextDocumentLayout }
	%struct.QTextDocumentConfig = type { %struct.QString }
	%struct.QTextDocumentPrivate = type { %struct.QObjectPrivate, %struct.QString, %"struct.QVector<QAbstractTextDocumentLayout::Selection>", i1, i32, i32, i1, i32, i32, i32, i32, i1, %struct.QTextFormatCollection, %struct.QTextBlockGroup*, %struct.QAbstractTextDocumentLayout*, %"struct.QFragmentMap<QTextBlockData>", %"struct.QFragmentMap<QTextBlockData>", i32, %"struct.QList<QPointer<QObject> >", %"struct.QList<QPointer<QObject> >", %"struct.QMap<QUrl,QVariant>", %"struct.QMap<QUrl,QVariant>", %"struct.QMap<QUrl,QVariant>", %struct.QTextDocumentConfig, i1, i1, %struct.QPointF }
	%struct.QTextFormat = type { %"struct.QSharedDataPointer<QTextFormatPrivate>", i32 }
	%struct.QTextFormatCollection = type { %"struct.QVector<QAbstractTextDocumentLayout::Selection>", %"struct.QVector<QAbstractTextDocumentLayout::Selection>", %"struct.QSet<uint>", %struct.QFont }
	%struct.QTextFormatPrivate = type opaque
	%"struct.QVector<QAbstractTextDocumentLayout::Selection>" = type { %struct..4._102 }
	%struct.QVectorData = type { %struct.QBasicAtomic, i32, i32, i8 }

define void @_ZNK13QTextDocument5printEP8QPrinter(%struct.QAbstractTextDocumentLayout* %this, %struct.QPrinter* %printer) {
entry:
	%tmp = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=2]
	%tmp.upgrd.1 = alloca %struct.QRectF, align 16		; <%struct.QRectF*> [#uses=5]
	%tmp2 = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=3]
	%tmp.upgrd.2 = alloca %struct.QFontMetrics, align 16		; <%struct.QFontMetrics*> [#uses=4]
	%tmp.upgrd.3 = alloca %struct.QFont, align 16		; <%struct.QFont*> [#uses=4]
	%tmp3 = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=2]
	%p = alloca %struct.QPainter, align 16		; <%struct.QPainter*> [#uses=14]
	%body = alloca %struct.QRectF, align 16		; <%struct.QRectF*> [#uses=9]
	%pageNumberPos = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=4]
	%scaledPageSize = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=6]
	%printerPageSize = alloca %struct.QPointF, align 16		; <%struct.QPointF*> [#uses=3]
	%fmt = alloca %struct.QTextBlockFormat, align 16		; <%struct.QTextBlockFormat*> [#uses=5]
	%font = alloca %struct.QFont, align 16		; <%struct.QFont*> [#uses=5]
	%tmp.upgrd.4 = call %struct.QTextDocumentPrivate* @_ZNK13QTextDocument6d_funcEv( %struct.QAbstractTextDocumentLayout* %this )		; <%struct.QTextDocumentPrivate*> [#uses=5]
	%tmp.upgrd.5 = getelementptr %struct.QPrinter* %printer, i32 0, i32 0		; <%struct.QPaintDevice*> [#uses=1]
	call void @_ZN8QPainterC1EP12QPaintDevice( %struct.QPainter* %p, %struct.QPaintDevice* %tmp.upgrd.5 )
	%tmp.upgrd.6 = invoke i1 @_ZNK8QPainter8isActiveEv( %struct.QPainter* %p )
			to label %invcont unwind label %cleanup329		; <i1> [#uses=1]
invcont:		; preds = %entry
	br i1 %tmp.upgrd.6, label %cond_next, label %cleanup328
cond_next:		; preds = %invcont
	%tmp8 = invoke %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument14documentLayoutEv( %struct.QAbstractTextDocumentLayout* %this )
			to label %invcont7 unwind label %cleanup329		; <%struct.QAbstractTextDocumentLayout*> [#uses=0]
invcont7:		; preds = %cond_next
	%tmp10 = getelementptr %struct.QTextDocumentPrivate* %tmp.upgrd.4, i32 0, i32 26		; <%struct.QPointF*> [#uses=1]
	call void @_ZN7QPointFC1Edd( %struct.QPointF* %tmp, double 0.000000e+00, double 0.000000e+00 )
	call void @_ZN6QRectFC1ERK7QPointFRK6QSizeF( %struct.QRectF* %body, %struct.QPointF* %tmp, %struct.QPointF* %tmp10 )
	call void @_ZN7QPointFC1Ev( %struct.QPointF* %pageNumberPos )
	%tmp12 = getelementptr %struct.QTextDocumentPrivate* %tmp.upgrd.4, i32 0, i32 26		; <%struct.QPointF*> [#uses=1]
	%tmp13 = call i1 @_ZNK6QSizeF7isValidEv( %struct.QPointF* %tmp12 )		; <i1> [#uses=1]
	br i1 %tmp13, label %cond_next15, label %bb
cond_next15:		; preds = %invcont7
	%tmp17 = getelementptr %struct.QTextDocumentPrivate* %tmp.upgrd.4, i32 0, i32 26		; <%struct.QPointF*> [#uses=1]
	%tmp.upgrd.7 = call double @_ZNK6QSizeF6heightEv( %struct.QPointF* %tmp17 )		; <double> [#uses=1]
	%tmp18 = fcmp oeq double %tmp.upgrd.7, 0x41DFFFFFFFC00000		; <i1> [#uses=1]
	br i1 %tmp18, label %bb, label %cond_next20
cond_next20:		; preds = %cond_next15
	br label %bb21
bb:		; preds = %cond_next15, %invcont7
	br label %bb21
bb21:		; preds = %bb, %cond_next20
	%iftmp.406.0 = phi i1 [ false, %bb ], [ true, %cond_next20 ]		; <i1> [#uses=1]
	br i1 %iftmp.406.0, label %cond_true24, label %cond_false
cond_true24:		; preds = %bb21
	%tmp.upgrd.8 = invoke i32 @_Z13qt_defaultDpiv( )
			to label %invcont25 unwind label %cleanup329		; <i32> [#uses=1]
invcont25:		; preds = %cond_true24
	%tmp26 = sitofp i32 %tmp.upgrd.8 to double		; <double> [#uses=2]
	%tmp30 = invoke %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument14documentLayoutEv( %struct.QAbstractTextDocumentLayout* %this )
			to label %invcont29 unwind label %cleanup329		; <%struct.QAbstractTextDocumentLayout*> [#uses=1]
invcont29:		; preds = %invcont25
	%tmp32 = invoke %struct.QPaintDevice* @_ZNK27QAbstractTextDocumentLayout11paintDeviceEv( %struct.QAbstractTextDocumentLayout* %tmp30 )
			to label %invcont31 unwind label %cleanup329		; <%struct.QPaintDevice*> [#uses=3]
invcont31:		; preds = %invcont29
	%tmp34 = icmp eq %struct.QPaintDevice* %tmp32, null		; <i1> [#uses=1]
	br i1 %tmp34, label %cond_next42, label %cond_true35
cond_true35:		; preds = %invcont31
	%tmp38 = invoke i32 @_ZNK12QPaintDevice11logicalDpiXEv( %struct.QPaintDevice* %tmp32 )
			to label %invcont37 unwind label %cleanup329		; <i32> [#uses=1]
invcont37:		; preds = %cond_true35
	%tmp38.upgrd.9 = sitofp i32 %tmp38 to double		; <double> [#uses=1]
	%tmp41 = invoke i32 @_ZNK12QPaintDevice11logicalDpiYEv( %struct.QPaintDevice* %tmp32 )
			to label %invcont40 unwind label %cleanup329		; <i32> [#uses=1]
invcont40:		; preds = %invcont37
	%tmp41.upgrd.10 = sitofp i32 %tmp41 to double		; <double> [#uses=1]
	br label %cond_next42
cond_next42:		; preds = %invcont40, %invcont31
	%sourceDpiY.2 = phi double [ %tmp41.upgrd.10, %invcont40 ], [ %tmp26, %invcont31 ]		; <double> [#uses=1]
	%sourceDpiX.2 = phi double [ %tmp38.upgrd.9, %invcont40 ], [ %tmp26, %invcont31 ]		; <double> [#uses=1]
	%tmp44 = getelementptr %struct.QPrinter* %printer, i32 0, i32 0		; <%struct.QPaintDevice*> [#uses=1]
	%tmp46 = invoke i32 @_ZNK12QPaintDevice11logicalDpiXEv( %struct.QPaintDevice* %tmp44 )
			to label %invcont45 unwind label %cleanup329		; <i32> [#uses=1]
invcont45:		; preds = %cond_next42
	%tmp46.upgrd.11 = sitofp i32 %tmp46 to double		; <double> [#uses=1]
	%tmp48 = fdiv double %tmp46.upgrd.11, %sourceDpiX.2		; <double> [#uses=2]
	%tmp50 = getelementptr %struct.QPrinter* %printer, i32 0, i32 0		; <%struct.QPaintDevice*> [#uses=1]
	%tmp52 = invoke i32 @_ZNK12QPaintDevice11logicalDpiYEv( %struct.QPaintDevice* %tmp50 )
			to label %invcont51 unwind label %cleanup329		; <i32> [#uses=1]
invcont51:		; preds = %invcont45
	%tmp52.upgrd.12 = sitofp i32 %tmp52 to double		; <double> [#uses=1]
	%tmp54 = fdiv double %tmp52.upgrd.12, %sourceDpiY.2		; <double> [#uses=2]
	invoke void @_ZN8QPainter5scaleEdd( %struct.QPainter* %p, double %tmp48, double %tmp54 )
			to label %invcont57 unwind label %cleanup329
invcont57:		; preds = %invcont51
	%tmp.upgrd.13 = getelementptr %struct.QPointF* %scaledPageSize, i32 0, i32 0		; <double*> [#uses=1]
	%tmp60 = getelementptr %struct.QTextDocumentPrivate* %tmp.upgrd.4, i32 0, i32 26, i32 0		; <double*> [#uses=1]
	%tmp61 = load double* %tmp60		; <double> [#uses=1]
	store double %tmp61, double* %tmp.upgrd.13
	%tmp62 = getelementptr %struct.QPointF* %scaledPageSize, i32 0, i32 1		; <double*> [#uses=1]
	%tmp63 = getelementptr %struct.QTextDocumentPrivate* %tmp.upgrd.4, i32 0, i32 26, i32 1		; <double*> [#uses=1]
	%tmp64 = load double* %tmp63		; <double> [#uses=1]
	store double %tmp64, double* %tmp62
	%tmp65 = call double* @_ZN6QSizeF6rwidthEv( %struct.QPointF* %scaledPageSize )		; <double*> [#uses=2]
	%tmp67 = load double* %tmp65		; <double> [#uses=1]
	%tmp69 = mul double %tmp67, %tmp48		; <double> [#uses=1]
	store double %tmp69, double* %tmp65
	%tmp71 = call double* @_ZN6QSizeF7rheightEv( %struct.QPointF* %scaledPageSize )		; <double*> [#uses=2]
	%tmp73 = load double* %tmp71		; <double> [#uses=1]
	%tmp75 = mul double %tmp73, %tmp54		; <double> [#uses=1]
	store double %tmp75, double* %tmp71
	%tmp78 = getelementptr %struct.QPrinter* %printer, i32 0, i32 0		; <%struct.QPaintDevice*> [#uses=1]
	%tmp80 = invoke i32 @_ZNK12QPaintDevice6heightEv( %struct.QPaintDevice* %tmp78 )
			to label %invcont79 unwind label %cleanup329		; <i32> [#uses=1]
invcont79:		; preds = %invcont57
	%tmp82 = getelementptr %struct.QPrinter* %printer, i32 0, i32 0		; <%struct.QPaintDevice*> [#uses=1]
	%tmp84 = invoke i32 @_ZNK12QPaintDevice5widthEv( %struct.QPaintDevice* %tmp82 )
			to label %invcont83 unwind label %cleanup329		; <i32> [#uses=1]
invcont83:		; preds = %invcont79
	%tmp80.upgrd.14 = sitofp i32 %tmp80 to double		; <double> [#uses=1]
	%tmp84.upgrd.15 = sitofp i32 %tmp84 to double		; <double> [#uses=1]
	call void @_ZN6QSizeFC1Edd( %struct.QPointF* %printerPageSize, double %tmp84.upgrd.15, double %tmp80.upgrd.14 )
	%tmp85 = call double @_ZNK6QSizeF6heightEv( %struct.QPointF* %printerPageSize )		; <double> [#uses=1]
	%tmp86 = call double @_ZNK6QSizeF6heightEv( %struct.QPointF* %scaledPageSize )		; <double> [#uses=1]
	%tmp87 = fdiv double %tmp85, %tmp86		; <double> [#uses=1]
	%tmp88 = call double @_ZNK6QSizeF5widthEv( %struct.QPointF* %printerPageSize )		; <double> [#uses=1]
	%tmp89 = call double @_ZNK6QSizeF5widthEv( %struct.QPointF* %scaledPageSize )		; <double> [#uses=1]
	%tmp90 = fdiv double %tmp88, %tmp89		; <double> [#uses=1]
	invoke void @_ZN8QPainter5scaleEdd( %struct.QPainter* %p, double %tmp90, double %tmp87 )
			to label %cond_next194 unwind label %cleanup329
cond_false:		; preds = %bb21
	%tmp.upgrd.16 = getelementptr %struct.QAbstractTextDocumentLayout* %this, i32 0, i32 0		; <%struct.QObject*> [#uses=1]
	%tmp95 = invoke %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument5cloneEP7QObject( %struct.QAbstractTextDocumentLayout* %this, %struct.QObject* %tmp.upgrd.16 )
			to label %invcont94 unwind label %cleanup329		; <%struct.QAbstractTextDocumentLayout*> [#uses=9]
invcont94:		; preds = %cond_false
	%tmp99 = invoke %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument14documentLayoutEv( %struct.QAbstractTextDocumentLayout* %tmp95 )
			to label %invcont98 unwind label %cleanup329		; <%struct.QAbstractTextDocumentLayout*> [#uses=1]
invcont98:		; preds = %invcont94
	%tmp101 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont100 unwind label %cleanup329		; <%struct.QPaintDevice*> [#uses=1]
invcont100:		; preds = %invcont98
	invoke void @_ZN27QAbstractTextDocumentLayout14setPaintDeviceEP12QPaintDevice( %struct.QAbstractTextDocumentLayout* %tmp99, %struct.QPaintDevice* %tmp101 )
			to label %invcont103 unwind label %cleanup329
invcont103:		; preds = %invcont100
	%tmp105 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont104 unwind label %cleanup329		; <%struct.QPaintDevice*> [#uses=1]
invcont104:		; preds = %invcont103
	%tmp107 = invoke i32 @_ZNK12QPaintDevice11logicalDpiYEv( %struct.QPaintDevice* %tmp105 )
			to label %invcont106 unwind label %cleanup329		; <i32> [#uses=1]
invcont106:		; preds = %invcont104
	%tmp108 = sitofp i32 %tmp107 to double		; <double> [#uses=1]
	%tmp109 = mul double %tmp108, 0x3FE93264C993264C		; <double> [#uses=1]
	%tmp109.upgrd.17 = fptosi double %tmp109 to i32		; <i32> [#uses=3]
	%tmp.upgrd.18 = call %struct.QTextBlockGroup* @_ZNK13QTextDocument9rootFrameEv( %struct.QAbstractTextDocumentLayout* %tmp95 )		; <%struct.QTextBlockGroup*> [#uses=1]
	invoke void @_ZNK10QTextFrame11frameFormatEv( %struct.QTextBlockFormat* sret  %fmt, %struct.QTextBlockGroup* %tmp.upgrd.18 )
			to label %invcont111 unwind label %cleanup329
invcont111:		; preds = %invcont106
	%tmp112 = sitofp i32 %tmp109.upgrd.17 to double		; <double> [#uses=1]
	invoke void @_ZN16QTextFrameFormat9setMarginEd( %struct.QTextBlockFormat* %fmt, double %tmp112 )
			to label %invcont114 unwind label %cleanup192
invcont114:		; preds = %invcont111
	%tmp116 = call %struct.QTextBlockGroup* @_ZNK13QTextDocument9rootFrameEv( %struct.QAbstractTextDocumentLayout* %tmp95 )		; <%struct.QTextBlockGroup*> [#uses=1]
	invoke void @_ZN10QTextFrame14setFrameFormatERK16QTextFrameFormat( %struct.QTextBlockGroup* %tmp116, %struct.QTextBlockFormat* %fmt )
			to label %invcont117 unwind label %cleanup192
invcont117:		; preds = %invcont114
	%tmp119 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont118 unwind label %cleanup192		; <%struct.QPaintDevice*> [#uses=1]
invcont118:		; preds = %invcont117
	%tmp121 = invoke i32 @_ZNK12QPaintDevice6heightEv( %struct.QPaintDevice* %tmp119 )
			to label %invcont120 unwind label %cleanup192		; <i32> [#uses=1]
invcont120:		; preds = %invcont118
	%tmp121.upgrd.19 = sitofp i32 %tmp121 to double		; <double> [#uses=1]
	%tmp123 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont122 unwind label %cleanup192		; <%struct.QPaintDevice*> [#uses=1]
invcont122:		; preds = %invcont120
	%tmp125 = invoke i32 @_ZNK12QPaintDevice5widthEv( %struct.QPaintDevice* %tmp123 )
			to label %invcont124 unwind label %cleanup192		; <i32> [#uses=1]
invcont124:		; preds = %invcont122
	%tmp125.upgrd.20 = sitofp i32 %tmp125 to double		; <double> [#uses=1]
	call void @_ZN6QRectFC1Edddd( %struct.QRectF* %tmp.upgrd.1, double 0.000000e+00, double 0.000000e+00, double %tmp125.upgrd.20, double %tmp121.upgrd.19 )
	%tmp126 = getelementptr %struct.QRectF* %body, i32 0, i32 0		; <double*> [#uses=1]
	%tmp127 = getelementptr %struct.QRectF* %tmp.upgrd.1, i32 0, i32 0		; <double*> [#uses=1]
	%tmp128 = load double* %tmp127		; <double> [#uses=1]
	store double %tmp128, double* %tmp126
	%tmp129 = getelementptr %struct.QRectF* %body, i32 0, i32 1		; <double*> [#uses=1]
	%tmp130 = getelementptr %struct.QRectF* %tmp.upgrd.1, i32 0, i32 1		; <double*> [#uses=1]
	%tmp131 = load double* %tmp130		; <double> [#uses=1]
	store double %tmp131, double* %tmp129
	%tmp132 = getelementptr %struct.QRectF* %body, i32 0, i32 2		; <double*> [#uses=1]
	%tmp133 = getelementptr %struct.QRectF* %tmp.upgrd.1, i32 0, i32 2		; <double*> [#uses=1]
	%tmp134 = load double* %tmp133		; <double> [#uses=1]
	store double %tmp134, double* %tmp132
	%tmp135 = getelementptr %struct.QRectF* %body, i32 0, i32 3		; <double*> [#uses=1]
	%tmp136 = getelementptr %struct.QRectF* %tmp.upgrd.1, i32 0, i32 3		; <double*> [#uses=1]
	%tmp137 = load double* %tmp136		; <double> [#uses=1]
	store double %tmp137, double* %tmp135
	%tmp138 = call double @_ZNK6QRectF6heightEv( %struct.QRectF* %body )		; <double> [#uses=1]
	%tmp139 = sitofp i32 %tmp109.upgrd.17 to double		; <double> [#uses=1]
	%tmp140 = sub double %tmp138, %tmp139		; <double> [#uses=1]
	%tmp142 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont141 unwind label %cleanup192		; <%struct.QPaintDevice*> [#uses=1]
invcont141:		; preds = %invcont124
	invoke void @_ZNK13QTextDocument11defaultFontEv( %struct.QFont* sret  %tmp.upgrd.3, %struct.QAbstractTextDocumentLayout* %tmp95 )
			to label %invcont144 unwind label %cleanup192
invcont144:		; preds = %invcont141
	invoke void @_ZN12QFontMetricsC1ERK5QFontP12QPaintDevice( %struct.QFontMetrics* %tmp.upgrd.2, %struct.QFont* %tmp.upgrd.3, %struct.QPaintDevice* %tmp142 )
			to label %invcont146 unwind label %cleanup173
invcont146:		; preds = %invcont144
	%tmp149 = invoke i32 @_ZNK12QFontMetrics6ascentEv( %struct.QFontMetrics* %tmp.upgrd.2 )
			to label %invcont148 unwind label %cleanup168		; <i32> [#uses=1]
invcont148:		; preds = %invcont146
	%tmp149.upgrd.21 = sitofp i32 %tmp149 to double		; <double> [#uses=1]
	%tmp150 = add double %tmp140, %tmp149.upgrd.21		; <double> [#uses=1]
	%tmp152 = invoke %struct.QPaintDevice* @_ZNK8QPainter6deviceEv( %struct.QPainter* %p )
			to label %invcont151 unwind label %cleanup168		; <%struct.QPaintDevice*> [#uses=1]
invcont151:		; preds = %invcont148
	%tmp154 = invoke i32 @_ZNK12QPaintDevice11logicalDpiYEv( %struct.QPaintDevice* %tmp152 )
			to label %invcont153 unwind label %cleanup168		; <i32> [#uses=1]
invcont153:		; preds = %invcont151
	%tmp155 = mul i32 %tmp154, 5		; <i32> [#uses=1]
	%tmp156 = sdiv i32 %tmp155, 72		; <i32> [#uses=1]
	%tmp156.upgrd.22 = sitofp i32 %tmp156 to double		; <double> [#uses=1]
	%tmp157 = add double %tmp150, %tmp156.upgrd.22		; <double> [#uses=1]
	%tmp158 = call double @_ZNK6QRectF5widthEv( %struct.QRectF* %body )		; <double> [#uses=1]
	%tmp159 = sitofp i32 %tmp109.upgrd.17 to double		; <double> [#uses=1]
	%tmp160 = sub double %tmp158, %tmp159		; <double> [#uses=1]
	call void @_ZN7QPointFC1Edd( %struct.QPointF* %tmp2, double %tmp160, double %tmp157 )
	%tmp161 = getelementptr %struct.QPointF* %pageNumberPos, i32 0, i32 0		; <double*> [#uses=1]
	%tmp162 = getelementptr %struct.QPointF* %tmp2, i32 0, i32 0		; <double*> [#uses=1]
	%tmp163 = load double* %tmp162		; <double> [#uses=1]
	store double %tmp163, double* %tmp161
	%tmp164 = getelementptr %struct.QPointF* %pageNumberPos, i32 0, i32 1		; <double*> [#uses=1]
	%tmp165 = getelementptr %struct.QPointF* %tmp2, i32 0, i32 1		; <double*> [#uses=1]
	%tmp166 = load double* %tmp165		; <double> [#uses=1]
	store double %tmp166, double* %tmp164
	invoke void @_ZN12QFontMetricsD1Ev( %struct.QFontMetrics* %tmp.upgrd.2 )
			to label %cleanup171 unwind label %cleanup173
cleanup168:		; preds = %invcont151, %invcont148, %invcont146
	invoke void @_ZN12QFontMetricsD1Ev( %struct.QFontMetrics* %tmp.upgrd.2 )
			to label %cleanup173 unwind label %cleanup173
cleanup171:		; preds = %invcont153
	invoke void @_ZN5QFontD1Ev( %struct.QFont* %tmp.upgrd.3 )
			to label %finally170 unwind label %cleanup192
cleanup173:		; preds = %cleanup168, %cleanup168, %invcont153, %invcont144
	invoke void @_ZN5QFontD1Ev( %struct.QFont* %tmp.upgrd.3 )
			to label %cleanup192 unwind label %cleanup192
finally170:		; preds = %cleanup171
	invoke void @_ZNK13QTextDocument11defaultFontEv( %struct.QFont* sret  %font, %struct.QAbstractTextDocumentLayout* %tmp95 )
			to label %invcont177 unwind label %cleanup192
invcont177:		; preds = %finally170
	invoke void @_ZN5QFont12setPointSizeEi( %struct.QFont* %font, i32 10 )
			to label %invcont179 unwind label %cleanup187
invcont179:		; preds = %invcont177
	invoke void @_ZN13QTextDocument14setDefaultFontERK5QFont( %struct.QAbstractTextDocumentLayout* %tmp95, %struct.QFont* %font )
			to label %invcont181 unwind label %cleanup187
invcont181:		; preds = %invcont179
	call void @_ZNK6QRectF4sizeEv( %struct.QPointF* sret  %tmp3, %struct.QRectF* %body )
	invoke void @_ZN13QTextDocument11setPageSizeERK6QSizeF( %struct.QAbstractTextDocumentLayout* %tmp95, %struct.QPointF* %tmp3 )
			to label %cleanup185 unwind label %cleanup187
cleanup185:		; preds = %invcont181
	invoke void @_ZN5QFontD1Ev( %struct.QFont* %font )
			to label %cleanup190 unwind label %cleanup192
cleanup187:		; preds = %invcont181, %invcont179, %invcont177
	invoke void @_ZN5QFontD1Ev( %struct.QFont* %font )
			to label %cleanup192 unwind label %cleanup192
cleanup190:		; preds = %cleanup185
	invoke void @_ZN16QTextFrameFormatD1Ev( %struct.QTextBlockFormat* %fmt )
			to label %cond_next194 unwind label %cleanup329
cleanup192:		; preds = %cleanup187, %cleanup187, %cleanup185, %finally170, %cleanup173, %cleanup173, %cleanup171, %invcont141, %invcont124, %invcont122, %invcont120, %invcont118, %invcont117, %invcont114, %invcont111
	invoke void @_ZN16QTextFrameFormatD1Ev( %struct.QTextBlockFormat* %fmt )
			to label %cleanup329 unwind label %cleanup329
cond_next194:		; preds = %cleanup190, %invcont83
	%clonedDoc.1 = phi %struct.QAbstractTextDocumentLayout* [ null, %invcont83 ], [ %tmp95, %cleanup190 ]		; <%struct.QAbstractTextDocumentLayout*> [#uses=3]
	%doc.1 = phi %struct.QAbstractTextDocumentLayout* [ %this, %invcont83 ], [ %tmp95, %cleanup190 ]		; <%struct.QAbstractTextDocumentLayout*> [#uses=2]
	%tmp197 = invoke i1 @_ZNK8QPrinter13collateCopiesEv( %struct.QPrinter* %printer )
			to label %invcont196 unwind label %cleanup329		; <i1> [#uses=1]
invcont196:		; preds = %cond_next194
	br i1 %tmp197, label %cond_true200, label %cond_false204
cond_true200:		; preds = %invcont196
	%tmp203 = invoke i32 @_ZNK8QPrinter9numCopiesEv( %struct.QPrinter* %printer )
			to label %invcont202 unwind label %cleanup329		; <i32> [#uses=1]
invcont202:		; preds = %cond_true200
	br label %cond_next208
cond_false204:		; preds = %invcont196
	%tmp207 = invoke i32 @_ZNK8QPrinter9numCopiesEv( %struct.QPrinter* %printer )
			to label %invcont206 unwind label %cleanup329		; <i32> [#uses=1]
invcont206:		; preds = %cond_false204
	br label %cond_next208
cond_next208:		; preds = %invcont206, %invcont202
	%pageCopies.0 = phi i32 [ %tmp203, %invcont202 ], [ 1, %invcont206 ]		; <i32> [#uses=2]
	%docCopies.0 = phi i32 [ 1, %invcont202 ], [ %tmp207, %invcont206 ]		; <i32> [#uses=2]
	%tmp211 = invoke i32 @_ZNK8QPrinter8fromPageEv( %struct.QPrinter* %printer )
			to label %invcont210 unwind label %cleanup329		; <i32> [#uses=3]
invcont210:		; preds = %cond_next208
	%tmp214 = invoke i32 @_ZNK8QPrinter6toPageEv( %struct.QPrinter* %printer )
			to label %invcont213 unwind label %cleanup329		; <i32> [#uses=3]
invcont213:		; preds = %invcont210
	%tmp216 = icmp eq i32 %tmp211, 0		; <i1> [#uses=1]
	br i1 %tmp216, label %cond_true217, label %cond_next225
cond_true217:		; preds = %invcont213
	%tmp219 = icmp eq i32 %tmp214, 0		; <i1> [#uses=1]
	br i1 %tmp219, label %cond_true220, label %cond_next225
cond_true220:		; preds = %cond_true217
	%tmp223 = invoke i32 @_ZNK13QTextDocument9pageCountEv( %struct.QAbstractTextDocumentLayout* %doc.1 )
			to label %invcont222 unwind label %cleanup329		; <i32> [#uses=1]
invcont222:		; preds = %cond_true220
	br label %cond_next225
cond_next225:		; preds = %invcont222, %cond_true217, %invcont213
	%toPage.1 = phi i32 [ %tmp223, %invcont222 ], [ %tmp214, %cond_true217 ], [ %tmp214, %invcont213 ]		; <i32> [#uses=2]
	%fromPage.1 = phi i32 [ 1, %invcont222 ], [ %tmp211, %cond_true217 ], [ %tmp211, %invcont213 ]		; <i32> [#uses=2]
	%tmp.page = invoke i32 @_ZNK8QPrinter9pageOrderEv( %struct.QPrinter* %printer )
			to label %invcont227 unwind label %cleanup329		; <i32> [#uses=1]
invcont227:		; preds = %cond_next225
	%tmp228 = icmp eq i32 %tmp.page, 1		; <i1> [#uses=1]
	br i1 %tmp228, label %cond_true230, label %cond_next234
cond_true230:		; preds = %invcont227
	br label %cond_next234
cond_next234:		; preds = %cond_true230, %invcont227
	%ascending.1 = phi i1 [ false, %cond_true230 ], [ true, %invcont227 ]		; <i1> [#uses=1]
	%toPage.2 = phi i32 [ %fromPage.1, %cond_true230 ], [ %toPage.1, %invcont227 ]		; <i32> [#uses=1]
	%fromPage.2 = phi i32 [ %toPage.1, %cond_true230 ], [ %fromPage.1, %invcont227 ]		; <i32> [#uses=1]
	br label %bb309
bb237:		; preds = %cond_true313, %cond_next293
	%iftmp.410.4 = phi i1 [ %iftmp.410.5, %cond_true313 ], [ %iftmp.410.1, %cond_next293 ]		; <i1> [#uses=1]
	%page.4 = phi i32 [ %fromPage.2, %cond_true313 ], [ %page.3, %cond_next293 ]		; <i32> [#uses=4]
	br label %bb273
invcont240:		; preds = %cond_true277
	%tmp242 = icmp eq i32 %tmp241, 2		; <i1> [#uses=1]
	br i1 %tmp242, label %bb252, label %cond_next244
cond_next244:		; preds = %invcont240
	%tmp247 = invoke i32 @_ZNK8QPrinter12printerStateEv( %struct.QPrinter* %printer )
			to label %invcont246 unwind label %cleanup329		; <i32> [#uses=1]
invcont246:		; preds = %cond_next244
	%tmp248 = icmp eq i32 %tmp247, 3		; <i1> [#uses=1]
	br i1 %tmp248, label %bb252, label %bb253
bb252:		; preds = %invcont246, %invcont240
	br label %bb254
bb253:		; preds = %invcont246
	br label %bb254
bb254:		; preds = %bb253, %bb252
	%iftmp.410.0 = phi i1 [ true, %bb252 ], [ false, %bb253 ]		; <i1> [#uses=2]
	br i1 %iftmp.410.0, label %UserCanceled, label %cond_next258
cond_next258:		; preds = %bb254
	invoke fastcc void @_Z9printPageiP8QPainterPK13QTextDocumentRK6QRectFRK7QPointF( i32 %page.4, %struct.QPainter* %p, %struct.QAbstractTextDocumentLayout* %doc.1, %struct.QRectF* %body, %struct.QPointF* %pageNumberPos )
			to label %invcont261 unwind label %cleanup329
invcont261:		; preds = %cond_next258
	%tmp263 = add i32 %pageCopies.0, -1		; <i32> [#uses=1]
	%tmp265 = icmp sgt i32 %tmp263, %j.4		; <i1> [#uses=1]
	br i1 %tmp265, label %cond_true266, label %cond_next270
cond_true266:		; preds = %invcont261
	%tmp269 = invoke i1 @_ZN8QPrinter7newPageEv( %struct.QPrinter* %printer )
			to label %cond_next270 unwind label %cleanup329		; <i1> [#uses=0]
cond_next270:		; preds = %cond_true266, %invcont261
	%tmp272 = add i32 %j.4, 1		; <i32> [#uses=1]
	br label %bb273
bb273:		; preds = %cond_next270, %bb237
	%iftmp.410.1 = phi i1 [ %iftmp.410.4, %bb237 ], [ %iftmp.410.0, %cond_next270 ]		; <i1> [#uses=2]
	%j.4 = phi i32 [ 0, %bb237 ], [ %tmp272, %cond_next270 ]		; <i32> [#uses=3]
	%tmp276 = icmp slt i32 %j.4, %pageCopies.0		; <i1> [#uses=1]
	br i1 %tmp276, label %cond_true277, label %bb280
cond_true277:		; preds = %bb273
	%tmp241 = invoke i32 @_ZNK8QPrinter12printerStateEv( %struct.QPrinter* %printer )
			to label %invcont240 unwind label %cleanup329		; <i32> [#uses=1]
bb280:		; preds = %bb273
	%tmp283 = icmp eq i32 %page.4, %toPage.2		; <i1> [#uses=1]
	br i1 %tmp283, label %bb297, label %cond_next285
cond_next285:		; preds = %bb280
	br i1 %ascending.1, label %cond_true287, label %cond_false290
cond_true287:		; preds = %cond_next285
	%tmp289 = add i32 %page.4, 1		; <i32> [#uses=1]
	br label %cond_next293
cond_false290:		; preds = %cond_next285
	%tmp292 = add i32 %page.4, -1		; <i32> [#uses=1]
	br label %cond_next293
cond_next293:		; preds = %cond_false290, %cond_true287
	%page.3 = phi i32 [ %tmp289, %cond_true287 ], [ %tmp292, %cond_false290 ]		; <i32> [#uses=1]
	%tmp296 = invoke i1 @_ZN8QPrinter7newPageEv( %struct.QPrinter* %printer )
			to label %bb237 unwind label %cleanup329		; <i1> [#uses=0]
bb297:		; preds = %bb280
	%tmp299 = add i32 %docCopies.0, -1		; <i32> [#uses=1]
	%tmp301 = icmp sgt i32 %tmp299, %i.1		; <i1> [#uses=1]
	br i1 %tmp301, label %cond_true302, label %cond_next306
cond_true302:		; preds = %bb297
	%tmp305 = invoke i1 @_ZN8QPrinter7newPageEv( %struct.QPrinter* %printer )
			to label %cond_next306 unwind label %cleanup329		; <i1> [#uses=0]
cond_next306:		; preds = %cond_true302, %bb297
	%tmp308 = add i32 %i.1, 1		; <i32> [#uses=1]
	br label %bb309
bb309:		; preds = %cond_next306, %cond_next234
	%iftmp.410.5 = phi i1 [ undef, %cond_next234 ], [ %iftmp.410.1, %cond_next306 ]		; <i1> [#uses=1]
	%i.1 = phi i32 [ 0, %cond_next234 ], [ %tmp308, %cond_next306 ]		; <i32> [#uses=3]
	%tmp312 = icmp slt i32 %i.1, %docCopies.0		; <i1> [#uses=1]
	br i1 %tmp312, label %cond_true313, label %UserCanceled
cond_true313:		; preds = %bb309
	br label %bb237
UserCanceled:		; preds = %bb309, %bb254
	%tmp318 = icmp eq %struct.QAbstractTextDocumentLayout* %clonedDoc.1, null		; <i1> [#uses=1]
	br i1 %tmp318, label %cleanup327, label %cond_true319
cond_true319:		; preds = %UserCanceled
	%tmp.upgrd.23 = getelementptr %struct.QAbstractTextDocumentLayout* %clonedDoc.1, i32 0, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	%tmp.upgrd.24 = load i32 (...)*** %tmp.upgrd.23		; <i32 (...)**> [#uses=1]
	%tmp322 = getelementptr i32 (...)** %tmp.upgrd.24, i32 4		; <i32 (...)**> [#uses=1]
	%tmp.upgrd.25 = load i32 (...)** %tmp322		; <i32 (...)*> [#uses=1]
	%tmp.upgrd.26 = bitcast i32 (...)* %tmp.upgrd.25 to void (%struct.QAbstractTextDocumentLayout*)*		; <void (%struct.QAbstractTextDocumentLayout*)*> [#uses=1]
	invoke void %tmp.upgrd.26( %struct.QAbstractTextDocumentLayout* %clonedDoc.1 )
			to label %cleanup327 unwind label %cleanup329
cleanup327:		; preds = %cond_true319, %UserCanceled
	call void @_ZN8QPainterD1Ev( %struct.QPainter* %p )
	ret void
cleanup328:		; preds = %invcont
	call void @_ZN8QPainterD1Ev( %struct.QPainter* %p )
	ret void
cleanup329:		; preds = %cond_true319, %cond_true302, %cond_next293, %cond_true277, %cond_true266, %cond_next258, %cond_next244, %cond_next225, %cond_true220, %invcont210, %cond_next208, %cond_false204, %cond_true200, %cond_next194, %cleanup192, %cleanup192, %cleanup190, %invcont106, %invcont104, %invcont103, %invcont100, %invcont98, %invcont94, %cond_false, %invcont83, %invcont79, %invcont57, %invcont51, %invcont45, %cond_next42, %invcont37, %cond_true35, %invcont29, %invcont25, %cond_true24, %cond_next, %entry
	call void @_ZN8QPainterD1Ev( %struct.QPainter* %p )
	unwind
}

declare void @_ZN6QSizeFC1Edd(%struct.QPointF*, double, double)

declare i1 @_ZNK6QSizeF7isValidEv(%struct.QPointF*)

declare double @_ZNK6QSizeF5widthEv(%struct.QPointF*)

declare double @_ZNK6QSizeF6heightEv(%struct.QPointF*)

declare double* @_ZN6QSizeF6rwidthEv(%struct.QPointF*)

declare double* @_ZN6QSizeF7rheightEv(%struct.QPointF*)

declare %struct.QTextDocumentPrivate* @_ZNK13QTextDocument6d_funcEv(%struct.QAbstractTextDocumentLayout*)

declare void @_ZN7QPointFC1Ev(%struct.QPointF*)

declare void @_ZN7QPointFC1Edd(%struct.QPointF*, double, double)

declare void @_ZN16QTextFrameFormat9setMarginEd(%struct.QTextBlockFormat*, double)

declare void @_ZN6QRectFC1Edddd(%struct.QRectF*, double, double, double, double)

declare void @_ZN6QRectFC1ERK7QPointFRK6QSizeF(%struct.QRectF*, %struct.QPointF*, %struct.QPointF*)

declare double @_ZNK6QRectF5widthEv(%struct.QRectF*)

declare double @_ZNK6QRectF6heightEv(%struct.QRectF*)

declare void @_ZNK6QRectF4sizeEv(%struct.QPointF*, %struct.QRectF*)

declare void @_ZN16QTextFrameFormatD1Ev(%struct.QTextBlockFormat*)

declare void @_ZNK10QTextFrame11frameFormatEv(%struct.QTextBlockFormat*, %struct.QTextBlockGroup*)

declare void @_ZN10QTextFrame14setFrameFormatERK16QTextFrameFormat(%struct.QTextBlockGroup*, %struct.QTextBlockFormat*)

declare i32 @_ZNK12QPaintDevice5widthEv(%struct.QPaintDevice*)

declare i32 @_ZNK12QPaintDevice6heightEv(%struct.QPaintDevice*)

declare i32 @_ZNK12QPaintDevice11logicalDpiXEv(%struct.QPaintDevice*)

declare i32 @_ZNK12QPaintDevice11logicalDpiYEv(%struct.QPaintDevice*)

declare %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument5cloneEP7QObject(%struct.QAbstractTextDocumentLayout*, %struct.QObject*)

declare void @_ZN5QFontD1Ev(%struct.QFont*)

declare %struct.QAbstractTextDocumentLayout* @_ZNK13QTextDocument14documentLayoutEv(%struct.QAbstractTextDocumentLayout*)

declare %struct.QTextBlockGroup* @_ZNK13QTextDocument9rootFrameEv(%struct.QAbstractTextDocumentLayout*)

declare i32 @_ZNK13QTextDocument9pageCountEv(%struct.QAbstractTextDocumentLayout*)

declare void @_ZNK13QTextDocument11defaultFontEv(%struct.QFont*, %struct.QAbstractTextDocumentLayout*)

declare void @_ZN13QTextDocument14setDefaultFontERK5QFont(%struct.QAbstractTextDocumentLayout*, %struct.QFont*)

declare void @_ZN13QTextDocument11setPageSizeERK6QSizeF(%struct.QAbstractTextDocumentLayout*, %struct.QPointF*)

declare void @_Z9printPageiP8QPainterPK13QTextDocumentRK6QRectFRK7QPointF(i32, %struct.QPainter*, %struct.QAbstractTextDocumentLayout*, %struct.QRectF*, %struct.QPointF*)

declare void @_ZN12QFontMetricsD1Ev(%struct.QFontMetrics*)

declare void @_ZN8QPainterC1EP12QPaintDevice(%struct.QPainter*, %struct.QPaintDevice*)

declare i1 @_ZNK8QPainter8isActiveEv(%struct.QPainter*)

declare i32 @_Z13qt_defaultDpiv()

declare %struct.QPaintDevice* @_ZNK27QAbstractTextDocumentLayout11paintDeviceEv(%struct.QAbstractTextDocumentLayout*)

declare void @_ZN8QPainter5scaleEdd(%struct.QPainter*, double, double)

declare %struct.QPaintDevice* @_ZNK8QPainter6deviceEv(%struct.QPainter*)

declare void @_ZN27QAbstractTextDocumentLayout14setPaintDeviceEP12QPaintDevice(%struct.QAbstractTextDocumentLayout*, %struct.QPaintDevice*)

declare void @_ZN12QFontMetricsC1ERK5QFontP12QPaintDevice(%struct.QFontMetrics*, %struct.QFont*, %struct.QPaintDevice*)

declare i32 @_ZNK12QFontMetrics6ascentEv(%struct.QFontMetrics*)

declare void @_ZN5QFont12setPointSizeEi(%struct.QFont*, i32)

declare i1 @_ZNK8QPrinter13collateCopiesEv(%struct.QPrinter*)

declare i32 @_ZNK8QPrinter9numCopiesEv(%struct.QPrinter*)

declare i32 @_ZNK8QPrinter8fromPageEv(%struct.QPrinter*)

declare i32 @_ZNK8QPrinter6toPageEv(%struct.QPrinter*)

declare i32 @_ZNK8QPrinter9pageOrderEv(%struct.QPrinter*)

declare i32 @_ZNK8QPrinter12printerStateEv(%struct.QPrinter*)

declare i1 @_ZN8QPrinter7newPageEv(%struct.QPrinter*)

declare void @_ZN8QPainterD1Ev(%struct.QPainter*)

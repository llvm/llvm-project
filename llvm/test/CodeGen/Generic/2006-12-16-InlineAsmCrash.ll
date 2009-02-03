; RUN: llvm-as < %s | llc -march=x86
; PR1049
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.QBasicAtomic = type { i32 }
	%struct.QByteArray = type { %"struct.QByteArray::Data"* }
	%"struct.QByteArray::Data" = type { %struct.QBasicAtomic, i32, i32, i8*, [1 x i8] }
	%struct.QFactoryLoader = type { %struct.QObject }
	%struct.QImageIOHandler = type { i32 (...)**, %struct.QImageIOHandlerPrivate* }
	%struct.QImageIOHandlerPrivate = type opaque
	%struct.QImageWriter = type { %struct.QImageWriterPrivate* }
	%struct.QImageWriterPrivate = type { %struct.QByteArray, %struct.QFactoryLoader*, i1, %struct.QImageIOHandler*, i32, float, %struct.QString, %struct.QString, i32, %struct.QString, %struct.QImageWriter* }
	%"struct.QList<QByteArray>" = type { %"struct.QList<QByteArray>::._20" }
	%"struct.QList<QByteArray>::._20" = type { %struct.QListData }
	%struct.QListData = type { %"struct.QListData::Data"* }
	%"struct.QListData::Data" = type { %struct.QBasicAtomic, i32, i32, i32, i8, [1 x i8*] }
	%struct.QObject = type { i32 (...)**, %struct.QObjectData* }
	%struct.QObjectData = type { i32 (...)**, %struct.QObject*, %struct.QObject*, %"struct.QList<QByteArray>", i8, [3 x i8], i32, i32 }
	%struct.QString = type { %"struct.QString::Data"* }
	%"struct.QString::Data" = type { %struct.QBasicAtomic, i32, i32, i16*, i8, i8, [1 x i16] }

define i1 @_ZNK12QImageWriter8canWriteEv() {
	%tmp62 = load %struct.QImageWriterPrivate** null		; <%struct.QImageWriterPrivate*> [#uses=1]
	%tmp = getelementptr %struct.QImageWriterPrivate* %tmp62, i32 0, i32 9		; <%struct.QString*> [#uses=1]
	%tmp75 = call %struct.QString* @_ZN7QStringaSERKS_( %struct.QString* %tmp, %struct.QString* null )		; <%struct.QString*> [#uses=0]
	call void asm sideeffect "lock\0Adecl $0\0Asetne 1", "=*m"( i32* null )
	ret i1 false
}

declare %struct.QString* @_ZN7QStringaSERKS_(%struct.QString*, %struct.QString*)

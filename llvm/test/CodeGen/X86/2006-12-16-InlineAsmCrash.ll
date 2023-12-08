; RUN: llc < %s
; PR1049
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.QBasicAtomic = type { i32 }
	%struct.QByteArray = type { ptr }
	%"struct.QByteArray::Data" = type { %struct.QBasicAtomic, i32, i32, ptr, [1 x i8] }
	%struct.QFactoryLoader = type { %struct.QObject }
	%struct.QImageIOHandler = type { ptr, ptr }
	%struct.QImageIOHandlerPrivate = type opaque
	%struct.QImageWriter = type { ptr }
	%struct.QImageWriterPrivate = type { %struct.QByteArray, ptr, i1, ptr, i32, float, %struct.QString, %struct.QString, i32, %struct.QString, ptr }
	%"struct.QList<QByteArray>" = type { %"struct.QList<QByteArray>::._20" }
	%"struct.QList<QByteArray>::._20" = type { %struct.QListData }
	%struct.QListData = type { ptr }
	%"struct.QListData::Data" = type { %struct.QBasicAtomic, i32, i32, i32, i8, [1 x ptr] }
	%struct.QObject = type { ptr, ptr }
	%struct.QObjectData = type { ptr, ptr, ptr, %"struct.QList<QByteArray>", i8, [3 x i8], i32, i32 }
	%struct.QString = type { ptr }
	%"struct.QString::Data" = type { %struct.QBasicAtomic, i32, i32, ptr, i8, i8, [1 x i16] }

define i1 @_ZNK12QImageWriter8canWriteEv() {
	%tmp62 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp = getelementptr %struct.QImageWriterPrivate, ptr %tmp62, i32 0, i32 9		; <ptr> [#uses=1]
	%tmp75 = call ptr @_ZN7QStringaSERKS_( ptr %tmp, ptr null )		; <ptr> [#uses=0]
	call void asm sideeffect "lock\0Adecl $0\0Asetne 1", "=*m"( ptr elementtype( i32) null )
	ret i1 false
}

declare ptr @_ZN7QStringaSERKS_(ptr, ptr)

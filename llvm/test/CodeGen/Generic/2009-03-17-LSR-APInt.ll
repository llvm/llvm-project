; RUN: llc < %s
; PR3806

; NVPTX does not support 'alias' yet
; XFAIL: target=nvptx{{.*}}

	%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
	%struct.Alignment = type { i32 }
	%struct.QDesignerFormWindowInterface = type { %struct.QWidget }
	%struct.QFont = type { ptr, i32 }
	%struct.QFontPrivate = type opaque
	%"struct.QHash<QString,QList<QAbstractExtensionFactory*> >" = type { %"struct.QHash<QString,QList<QAbstractExtensionFactory*> >::._120" }
	%"struct.QHash<QString,QList<QAbstractExtensionFactory*> >::._120" = type { ptr }
	%struct.QHashData = type { ptr, ptr, %struct.Alignment, i32, i32, i16, i16, i32, i8 }
	%"struct.QHashData::Node" = type { ptr, i32 }
	%"struct.QList<QAbstractExtensionFactory*>" = type { %"struct.QList<QAbstractExtensionFactory*>::._101" }
	%"struct.QList<QAbstractExtensionFactory*>::._101" = type { %struct.QListData }
	%struct.QListData = type { ptr }
	%"struct.QListData::Data" = type { %struct.Alignment, i32, i32, i32, i8, [1 x ptr] }
	%struct.QObject = type { ptr, ptr }
	%struct.QObjectData = type { ptr, ptr, ptr, %"struct.QList<QAbstractExtensionFactory*>", i32, i32 }
	%struct.QPaintDevice.base = type { ptr, i16 }
	%"struct.QPair<int,int>" = type { i32, i32 }
	%struct.QPalette = type { ptr, i32 }
	%struct.QPalettePrivate = type opaque
	%struct.QRect = type { i32, i32, i32, i32 }
	%struct.QWidget = type { %struct.QObject, %struct.QPaintDevice.base, ptr }
	%struct.QWidgetData = type { i64, i32, %struct.Alignment, i8, i8, i16, %struct.QRect, %struct.QPalette, %struct.QFont, %struct.QRect }
	%struct.__pthread_list_t = type { ptr, ptr }
	%struct.pthread_attr_t = type { i64, [48 x i8] }
	%struct.pthread_mutex_t = type { %struct..0__pthread_mutex_s }
	%"struct.qdesigner_internal::Grid" = type { i32, i32, ptr, ptr, ptr }
	%"struct.qdesigner_internal::GridLayout" = type { %"struct.qdesigner_internal::Layout", %"struct.QPair<int,int>", ptr }
	%"struct.qdesigner_internal::Layout" = type { %struct.QObject, %"struct.QList<QAbstractExtensionFactory*>", ptr, %"struct.QHash<QString,QList<QAbstractExtensionFactory*> >", ptr, ptr, i8, %"struct.QPair<int,int>", %struct.QRect, i8 }

@_ZL20__gthrw_pthread_oncePiPFvvE = weak alias i32 (ptr, ptr), ptr @pthread_once		; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = weak alias ptr (i32), ptr @pthread_getspecific		; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = weak alias i32 (i32, ptr), ptr @pthread_setspecific		; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = weak alias i32 (ptr, ptr, ptr, ptr), ptr @pthread_create		; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = weak alias i32 (i64), ptr @pthread_cancel		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_lock		; <ptr> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_trylock		; <ptr> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_unlock		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = weak alias i32 (ptr, ptr), ptr @pthread_mutex_init		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = weak alias i32 (ptr, ptr), ptr @pthread_key_create		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = weak alias i32 (i32), ptr @pthread_key_delete		; <ptr> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_init		; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = weak alias i32 (ptr, i32), ptr @pthread_mutexattr_settype		; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_destroy		; <ptr> [#uses=0]

define void @_ZN18qdesigner_internal10GridLayout9buildGridEv(ptr %this) nounwind {
entry:
	br label %bb44

bb44:		; preds = %bb47, %entry
	%indvar = phi i128 [ %indvar.next144, %bb47 ], [ 0, %entry ]		; <i128> [#uses=2]
	br i1 false, label %bb46, label %bb47

bb46:		; preds = %bb44
	%tmp = shl i128 %indvar, 64		; <i128> [#uses=1]
	%tmp96 = and i128 %tmp, 79228162495817593519834398720		; <i128> [#uses=0]
	br label %bb47

bb47:		; preds = %bb46, %bb44
	%indvar.next144 = add i128 %indvar, 1		; <i128> [#uses=1]
	br label %bb44
}

define i32 @pthread_once(ptr, ptr) addrspace(0) {
  ret i32 0
}

define ptr @pthread_getspecific(i32) addrspace(0) {
  ret ptr null
}

define i32 @pthread_setspecific(i32, ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_create(ptr, ptr, ptr, ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_cancel(i64) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutex_lock(ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutex_trylock(ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutex_unlock(ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutex_init(ptr, ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_key_create(ptr, ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_key_delete(i32) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutexattr_init(ptr) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutexattr_settype(ptr, i32) addrspace(0) {
  ret i32 0
}

define i32 @pthread_mutexattr_destroy(ptr) addrspace(0) {
  ret i32 0
}

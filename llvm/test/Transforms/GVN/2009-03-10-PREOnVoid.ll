; RUN: opt < %s -passes=gvn -disable-output
; PR3775

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%"struct.__gnu_cxx::hash<ptr>" = type <{ i8 }>
	%struct.__sched_param = type { i32 }
	%struct._pthread_descr_struct = type opaque
	%struct.pthread_attr_t = type { i32, i32, %struct.__sched_param, i32, i32, i32, i32, ptr, i32 }
	%struct.pthread_mutex_t = type { i32, i32, ptr, i32, %llvm.dbg.anchor.type }
	%"struct.std::_Rb_tree<ptr,std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > >,std::_Select1st<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >,std::less<ptr>,std::allocator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > > >" = type { %"struct.std::_Rb_tree<ptr,std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > >,std::_Select1st<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >,std::less<ptr>,std::allocator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > > >::_Rb_tree_impl<std::less<ptr>,false>" }
	%"struct.std::_Rb_tree<ptr,std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > >,std::_Select1st<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >,std::less<ptr>,std::allocator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > > >::_Rb_tree_impl<std::less<ptr>,false>" = type { %"struct.__gnu_cxx::hash<ptr>", %"struct.std::_Rb_tree_node_base", i32 }
	%"struct.std::_Rb_tree_iterator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >" = type { ptr }
	%"struct.std::_Rb_tree_node_base" = type { i32, ptr, ptr, ptr }
	%"struct.std::pair<std::_Rb_tree_iterator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >,bool>" = type { %"struct.std::_Rb_tree_iterator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >", i8 }
	%"struct.std::pair<ptr const,ptr>" = type { ptr, ptr }

@_ZL20__gthrw_pthread_oncePiPFvvE = weak alias i32 (ptr, ptr), ptr @pthread_once		; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = weak alias ptr (i32), ptr @pthread_getspecific		; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = weak alias i32 (i32, ptr), ptr @pthread_setspecific		; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK16__pthread_attr_sPFPvS3_ES3_ = weak alias i32 (ptr, ptr, ptr, ptr), ptr @pthread_create		; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = weak alias i32 (i32), ptr @pthread_cancel		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_lock		; <ptr> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_trylock		; <ptr> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_unlock		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = weak alias i32 (ptr, ptr), ptr @pthread_mutex_init		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = weak alias i32 (ptr, ptr), ptr @pthread_key_create		; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = weak alias i32 (i32), ptr @pthread_key_delete		; <ptr> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_init		; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = weak alias i32 (ptr, i32), ptr @pthread_mutexattr_settype		; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_destroy		; <ptr> [#uses=0]

declare fastcc void @_ZNSt10_Select1stISt4pairIKPvS1_EEC1Ev() nounwind readnone

define fastcc void @_ZNSt8_Rb_treeIPvSt4pairIKS0_S0_ESt10_Select1stIS3_ESt4lessIS0_ESaIS3_EE16_M_insert_uniqueERKS3_(ptr noalias nocapture sret(%"struct.std::pair<std::_Rb_tree_iterator<std::pair<ptr const, std::vector<ShadowInfo, std::allocator<ShadowInfo> > > >,bool>") %agg.result, ptr %this, ptr %__v) nounwind {
entry:
	br i1 false, label %bb7, label %bb

bb:		; preds = %bb, %entry
	br i1 false, label %bb5, label %bb

bb5:		; preds = %bb
	call fastcc void @_ZNSt10_Select1stISt4pairIKPvS1_EEC1Ev() nounwind
	br i1 false, label %bb11, label %bb7

bb7:		; preds = %bb5, %entry
	br label %bb11

bb11:		; preds = %bb7, %bb5
	call fastcc void @_ZNSt10_Select1stISt4pairIKPvS1_EEC1Ev() nounwind
	unreachable
}

define i32 @pthread_once(ptr, ptr) {
       ret i32 0
}

define ptr @pthread_getspecific(i32) {
       ret ptr null
}

define i32 @pthread_setspecific(i32, ptr) {
        ret i32 0
}

define i32 @pthread_create(ptr, ptr, ptr, ptr) {
       ret i32 0
}

define i32 @pthread_cancel(i32) {
      ret i32 0
}

define i32 @pthread_mutex_lock(ptr) {
       ret i32 0
}

define i32 @pthread_mutex_trylock(ptr) {
       ret i32 0
}

define i32 @pthread_mutex_unlock(ptr) {
       ret i32 0
}

define i32 @pthread_mutex_init(ptr, ptr) {
        ret i32 0
}

define i32 @pthread_key_create(ptr, ptr) {
       ret i32 0
}

define i32 @pthread_key_delete(i32) {
        ret i32 0
}

define i32 @pthread_mutexattr_init(ptr) {
        ret i32 0
}

define i32 @pthread_mutexattr_settype(ptr, i32) {
        ret i32 0
}

define i32 @pthread_mutexattr_destroy(ptr) {
       ret i32 0
}

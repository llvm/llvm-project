; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
%struct.__pthread_list_t = type { %struct.__pthread_list_t*, %struct.__pthread_list_t* }
%union.pthread_attr_t = type { i64, [12 x i32] }
%union.pthread_mutex_t = type { %struct..0__pthread_mutex_s }
%union.pthread_mutexattr_t = type { i32 }

@_ZL20__gthrw_pthread_oncePiPFvvE = weak alias i32 (i32*, void ()*), i32 (i32*, void ()*)* @pthread_once ; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = weak alias i8* (i32), i8* (i32)* @pthread_getspecific ; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = weak alias i32 (i32, i8*), i32 (i32, i8*)* @pthread_setspecific ; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = weak alias i32 (i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*), i32 (i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create ; <i32 (i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = weak alias i32 (i64), i32 (i64)* @pthread_cancel ; <i32 (i64)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*), i32 (%union.pthread_mutex_t*)* @pthread_mutex_lock ; <i32 (%union.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*), i32 (%union.pthread_mutex_t*)* @pthread_mutex_trylock ; <i32 (%union.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*), i32 (%union.pthread_mutex_t*)* @pthread_mutex_unlock ; <i32 (%union.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutex_t*, %union.pthread_mutexattr_t*), i32 (%union.pthread_mutex_t*, %union.pthread_mutexattr_t*)* @pthread_mutex_init ; <i32 (%union.pthread_mutex_t*, %union.pthread_mutexattr_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = weak alias i32 (i32*, void (i8*)*), i32 (i32*, void (i8*)*)* @pthread_key_create ; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = weak alias i32 (i32), i32 (i32)* @pthread_key_delete ; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutexattr_t*), i32 (%union.pthread_mutexattr_t*)* @pthread_mutexattr_init ; <i32 (%union.pthread_mutexattr_t*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = weak alias i32 (%union.pthread_mutexattr_t*, i32), i32 (%union.pthread_mutexattr_t*, i32)* @pthread_mutexattr_settype ; <i32 (%union.pthread_mutexattr_t*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutexattr_t*), i32 (%union.pthread_mutexattr_t*)* @pthread_mutexattr_destroy ; <i32 (%union.pthread_mutexattr_t*)*> [#uses=0]

define void @_ZL6createP6node_tii3v_tS1_d() {
entry:
  br i1 undef, label %bb, label %bb5

bb:                                               ; preds = %entry
  br i1 false, label %bb1, label %bb3

bb1:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb1, %bb
  %iftmp.99.0 = phi i64 [ undef, %bb1 ], [ 1, %bb ] ; <i64> [#uses=0]
  br label %bb5

bb5:                                              ; preds = %bb3, %entry
  br i1 undef, label %return, label %bb7

bb7:                                              ; preds = %bb5
  unreachable

return:                                           ; preds = %bb5
  ret void
}

define i32 @pthread_once(i32*, void ()*) {
  ret i32 0
}

define i8* @pthread_getspecific(i32) {
  ret i8* null
}

define i32 @pthread_setspecific(i32, i8*) {
  ret i32 0
}

define i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*) {
  ret i32 0
}

define i32 @pthread_cancel(i64) {
  ret i32 0
}

define i32 @pthread_mutex_lock(%union.pthread_mutex_t*) {
  ret i32 0
}

define i32 @pthread_mutex_trylock(%union.pthread_mutex_t*) {
  ret i32 0
}

define i32 @pthread_mutex_unlock(%union.pthread_mutex_t*) {
  ret i32 0
}

define i32 @pthread_mutex_init(%union.pthread_mutex_t*, %union.pthread_mutexattr_t*) {
  ret i32 0
}

define i32 @pthread_key_create(i32*, void (i8*)*) {
  ret i32 0
}

define i32 @pthread_key_delete(i32) {
  ret i32 0
}

define i32 @pthread_mutexattr_init(%union.pthread_mutexattr_t*) {
  ret i32 0
}

define i32 @pthread_mutexattr_settype(%union.pthread_mutexattr_t*, i32) {
  ret i32 0
}

define i32 @pthread_mutexattr_destroy(%union.pthread_mutexattr_t*) {
  ret i32 0
}

; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
%struct.__pthread_list_t = type { ptr, ptr }
%union.pthread_attr_t = type { i64, [12 x i32] }
%union.pthread_mutex_t = type { %struct..0__pthread_mutex_s }
%union.pthread_mutexattr_t = type { i32 }

@_ZL20__gthrw_pthread_oncePiPFvvE = weak alias i32 (ptr, ptr), ptr @pthread_once ; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = weak alias ptr (i32), ptr @pthread_getspecific ; <ptr> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = weak alias i32 (i32, ptr), ptr @pthread_setspecific ; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = weak alias i32 (ptr, ptr, ptr, ptr), ptr @pthread_create ; <ptr> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = weak alias i32 (i64), ptr @pthread_cancel ; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_lock ; <ptr> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_trylock ; <ptr> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = weak alias i32 (ptr), ptr @pthread_mutex_unlock ; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = weak alias i32 (ptr, ptr), ptr @pthread_mutex_init ; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = weak alias i32 (ptr, ptr), ptr @pthread_key_create ; <ptr> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = weak alias i32 (i32), ptr @pthread_key_delete ; <ptr> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_init ; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = weak alias i32 (ptr, i32), ptr @pthread_mutexattr_settype ; <ptr> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = weak alias i32 (ptr), ptr @pthread_mutexattr_destroy ; <ptr> [#uses=0]

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

define i32 @pthread_cancel(i64) {
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

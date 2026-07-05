; RUN: opt < %s -passes=globalopt -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
        %struct.SFLMutableListItem = type { i16 }
        %struct.__CFDictionary = type opaque
        %struct.__CFString = type opaque
        %struct.__builtin_CFString = type { ptr, i32, ptr, i32 }
@_ZZ19SFLGetVisibilityKeyvE19_kSFLLVisibilityKey = internal global ptr null             ; <ptr> [#uses=2]
@_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey = internal global ptr null               ; <ptr> [#uses=7]
@0 = internal constant %struct.__builtin_CFString {
    ptr @__CFConstantStringClassReference,
    i32 1992,
    ptr @.str,
    i32 13 }, section "__DATA,__cfstring"               ; <ptr>:0 [#uses=1]
@__CFConstantStringClassReference = external global [0 x i32]           ; <ptr> [#uses=1]
@.str = internal constant [14 x i8] c"AlwaysVisible\00"         ; <ptr> [#uses=1]
@_ZZ21SFLGetNeverVisibleKeyvE21_kSFLLNeverVisibleKey = internal global ptr null         ; <ptr> [#uses=2]

define ptr @_Z19SFLGetVisibilityKeyv() {
entry:
        %tmp1 = load ptr, ptr @_ZZ19SFLGetVisibilityKeyvE19_kSFLLVisibilityKey              ; <ptr> [#uses=1]
        ret ptr %tmp1
}

define ptr @_Z22SFLGetAlwaysVisibleKeyv() {
entry:
        %tmp1 = load ptr, ptr @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey                ; <ptr> [#uses=1]
        %tmp2 = icmp eq ptr %tmp1, null         ; <i1> [#uses=1]
        br i1 %tmp2, label %cond_true, label %cond_next

cond_true:              ; preds = %entry
        store ptr @0, ptr @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey
        br label %cond_next

cond_next:              ; preds = %entry, %cond_true
        %tmp4 = load ptr, ptr @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey                ; <ptr> [#uses=1]
        ret ptr %tmp4
}

define ptr @_Z21SFLGetNeverVisibleKeyv() {
entry:
        %tmp1 = load ptr, ptr @_ZZ21SFLGetNeverVisibleKeyvE21_kSFLLNeverVisibleKey          ; <ptr> [#uses=1]
        ret ptr %tmp1
}

define ptr @_ZN18SFLMutableListItem18GetPrefsDictionaryEv(ptr %this) {
entry:
        %tmp4 = getelementptr %struct.SFLMutableListItem, ptr %this, i32 0, i32 0  ; <ptr> [#uses=1]
        %tmp5 = load i16, ptr %tmp4         ; <i16> [#uses=1]
        %tmp6 = icmp eq i16 %tmp5, 0            ; <i1> [#uses=1]
        br i1 %tmp6, label %cond_next22, label %cond_true

cond_true:              ; preds = %entry
        %tmp9 = load ptr, ptr @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey                ; <ptr> [#uses=1]
        %tmp10 = icmp eq ptr %tmp9, null                ; <i1> [#uses=1]
        br i1 %tmp10, label %cond_true13, label %cond_next22

cond_true13:            ; preds = %cond_true
        store ptr @0, ptr @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey
        br label %cond_next22

cond_next22:            ; preds = %entry, %cond_true13, %cond_true
        %iftmp.1.0.in = phi ptr [ @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey, %cond_true ], [ @_ZZ22SFLGetAlwaysVisibleKeyvE22_kSFLLAlwaysVisibleKey, %cond_true13 ], [ @_ZZ21SFLGetNeverVisibleKeyvE21_kSFLLNeverVisibleKey, %entry ]             ; <ptr> [#uses=1]
        %iftmp.1.0 = load ptr, ptr %iftmp.1.0.in            ; <ptr> [#uses=1]
        %tmp24 = load ptr, ptr @_ZZ19SFLGetVisibilityKeyvE19_kSFLLVisibilityKey             ; <ptr> [#uses=1]
        call void @_Z20CFDictionaryAddValuePKvS0_( ptr %tmp24, ptr %iftmp.1.0 )
        ret ptr undef
}

declare void @_Z20CFDictionaryAddValuePKvS0_(ptr, ptr)


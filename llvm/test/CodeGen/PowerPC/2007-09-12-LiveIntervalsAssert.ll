; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin

declare void @cxa_atexit_check_1(i8*)

define i32 @check_cxa_atexit(i32 (void (i8*)*, i8*, i8*)* %cxa_atexit, void (i8*)* %cxa_finalize) {
entry:
        %tmp7 = call i32 null( void (i8*)* @cxa_atexit_check_1, i8* null, i8* null )            ; <i32> [#uses=0]
        br i1 false, label %cond_true, label %cond_next

cond_true:    ; preds = %entry
        ret i32 0

cond_next:        ; preds = %entry
        ret i32 0
}

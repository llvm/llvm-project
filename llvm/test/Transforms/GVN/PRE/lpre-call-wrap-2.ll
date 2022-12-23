; RUN: opt -S -passes=gvn -enable-load-pre < %s | FileCheck %s
;
; The partially redundant load in bb1 should be hoisted to "bb".  This comes
; from this C code (GCC PR 23455):
;   unsigned outcnt;  extern void flush_outbuf(void);
;   void bi_windup(unsigned char *outbuf, unsigned char bi_buf) {
;     outbuf[outcnt] = bi_buf;
;     if (outcnt == 16384)
;       flush_outbuf();
;     outbuf[outcnt] = bi_buf;
;   }
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
@outcnt = common global i32 0		; <ptr> [#uses=3]

define void @bi_windup(ptr %outbuf, i8 zeroext %bi_buf) nounwind {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%0 = load i32, ptr @outcnt, align 4		; <i32> [#uses=1]
	%1 = getelementptr i8, ptr %outbuf, i32 %0		; <ptr> [#uses=1]
	store i8 %bi_buf, ptr %1, align 1
	%2 = load i32, ptr @outcnt, align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 16384		; <i1> [#uses=1]
	br i1 %3, label %bb, label %bb1

bb:		; preds = %entry
	call void @flush_outbuf() nounwind
	br label %bb1

bb1:		; preds = %bb, %entry
; CHECK: bb1:
; CHECK-NEXT: phi
; CHECK-NEXT: getelementptr
	%4 = load i32, ptr @outcnt, align 4		; <i32> [#uses=1]
	%5 = getelementptr i8, ptr %outbuf, i32 %4		; <ptr> [#uses=1]
	store i8 %bi_buf, ptr %5, align 1
	ret void
}

declare void @flush_outbuf()

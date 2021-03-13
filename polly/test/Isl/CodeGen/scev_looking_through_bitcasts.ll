; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Scalar write of bitcasted value. Instead of writing %b of type
; %structty, the SCEV expression looks through the bitcast such that
; SCEVExpander returns %add.ptr81.i of type i8* to be the new value
; of %b.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%structty = type { %structty*, %structty*, i32, [2 x i64] }

define void @bitmap_set_range() {
entry:
  %a = ptrtoint i8* undef to i64
  br label %cond.end32.i

cond.end32.i:
  br i1 false, label %cond.true67.i, label %cond.end73.i

cond.true67.i:
  br label %cond.end73.i

cond.end73.i:
  %add.ptr81.i = getelementptr inbounds i8, i8* null, i64 %a
  %b = bitcast i8* %add.ptr81.i to %structty*
  br label %bitmap_element_allocate.exit

bitmap_element_allocate.exit:
  %tobool43 = icmp eq %structty* %b, null
  ret void
}


; CHECK:       polly.stmt.cond.end73.i:
; CHECK-NEXT:   %0 = bitcast %structty** %b.s2a to i8**
; CHECK-NEXT:   store i8* undef, i8** %0
; CHECK-NEXT:   br label %polly.exiting

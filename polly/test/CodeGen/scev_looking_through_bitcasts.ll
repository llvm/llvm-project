; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; Scalar write of bitcasted value. Instead of writing %b of type
; %structty, the SCEV expression looks through the bitcast such that
; SCEVExpander returns %add.ptr81.i of type i8* to be the new value
; of %b.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%structty = type { ptr, ptr, i32, [2 x i64] }

define void @bitmap_set_range() {
entry:
  %a = ptrtoint ptr undef to i64
  br label %cond.end32.i

cond.end32.i:
  br i1 false, label %cond.true67.i, label %cond.end73.i

cond.true67.i:
  br label %cond.end73.i

cond.end73.i:
  %add.ptr81.i = getelementptr inbounds i8, ptr null, i64 %a
  br label %bitmap_element_allocate.exit

bitmap_element_allocate.exit:
  %tobool43 = icmp eq ptr %add.ptr81.i, null
  ret void
}



; CHECK:      polly.stmt.cond.end73.i:
; CHECK-NEXT:   %scevgep = getelementptr i8, ptr null, i64 %a
; CHECK-NEXT:   store ptr %scevgep, ptr %add.ptr81.i.s2a, align 8
; CHECK-NEXT:   br label %polly.exiting

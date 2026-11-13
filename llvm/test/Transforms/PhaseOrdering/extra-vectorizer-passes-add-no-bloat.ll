; RUN: opt -O3 -extra-vectorizer-passes=false -S %s 2>&1 | FileCheck %s --check-prefix=WITHOUT
; RUN: opt -O3 -extra-vectorizer-passes -S %s 2>&1 | FileCheck %s --check-prefix=WITH
;
; This test checks that -extra-vectorizer-passes does not cause excessive
; outer loop unrolling that duplicates vectorized inner loops without
; measurable performance benefit.
;
; When performance is equivalent, the compiler should prefer smaller code.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @nested_loop_inner_vectorized(ptr nocapture noundef %buf, i64 noundef %width, i64 noundef %height, i64 noundef %stride) {
entry:
  %cmp.height = icmp eq i64 %height, 0
  br i1 %cmp.height, label %exit, label %outer.ph

outer.ph:
  %cmp.width = icmp eq i64 %width, 0
  br label %outer.header

outer.header:
  %y = phi i64 [ 0, %outer.ph ], [ %y.next, %outer.latch ]
  %buf.y = phi ptr [ %buf, %outer.ph ], [ %buf.next, %outer.latch ]
  br i1 %cmp.width, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %x = phi i64 [ 0, %inner.ph ], [ %x.next, %inner ]
  %ptr = getelementptr inbounds i8, ptr %buf.y, i64 %x
  %val = load i8, ptr %ptr, align 1
  %inc = add i8 %val, 1
  store i8 %inc, ptr %ptr, align 1
  %x.next = add nuw i64 %x, 1
  %cmp.x = icmp eq i64 %x.next, %width
  br i1 %cmp.x, label %outer.latch, label %inner

outer.latch:
  %buf.next = getelementptr inbounds i8, ptr %buf.y, i64 %stride
  %y.next = add nuw i64 %y, 1
  %cmp.y = icmp eq i64 %y.next, %height
  br i1 %cmp.y, label %exit, label %outer.header

exit:
  ret void
}

; Original test case from https://github.com/llvm/llvm-project/issues/203904

define void @ass_shift_bitmap_c(ptr noalias nocapture %buf, i64 %stride, i64 %width, i64 %height, i32 %shift_x, i32 %shift_y, ptr noalias nocapture %tmp) {
entry:
  %cmp.height = icmp eq i64 %height, 0
  br i1 %cmp.height, label %exit, label %outer.ph

outer.ph:
  %cmp.width = icmp eq i64 %width, 0
  br label %outer.header

outer.header:
  %y = phi i64 [ 0, %outer.ph ], [ %y.next, %outer.latch ]
  %buf.y = phi ptr [ %buf, %outer.ph ], [ %buf.next, %outer.latch ]
  br i1 %cmp.width, label %outer.latch, label %inner.ph

inner.ph:
  br label %inner

inner:
  %x = phi i64 [ 0, %inner.ph ], [ %x.next, %inner ]
  %shifted_from_left = phi i32 [ 0, %inner.ph ], [ %b_x, %inner ]
  ; px = buf[x] << 1
  %buf.ptr = getelementptr inbounds i8, ptr %buf.y, i64 %x
  %buf.val = load i8, ptr %buf.ptr, align 1
  %buf.ext = zext i8 %buf.val to i32
  %px.init = shl nuw nsw i32 %buf.ext, 1
  ; shifted_from_top = tmp[x]
  %tmp.ptr = getelementptr inbounds i16, ptr %tmp, i64 %x
  %shifted_from_top = load i16, ptr %tmp.ptr, align 2
  ; b_x = px * shift_x >> 6
  %mul.x = mul nuw nsw i32 %px.init, %shift_x
  %b_x = lshr i32 %mul.x, 6
  ; px += shifted_from_left - b_x
  %px.add.left = add nuw nsw i32 %px.init, %shifted_from_left
  %px.sub.bx = sub nsw i32 %px.add.left, %b_x
  %px.masked = and i32 %px.sub.bx, 65535
  ; b_y = px * shift_y >> 6
  %mul.y = mul nuw nsw i32 %px.masked, %shift_y
  %b_y.full = lshr i32 %mul.y, 6
  %b_y = trunc i32 %b_y.full to i16
  ; px += shifted_from_top - b_y
  %px.trunc = trunc i32 %px.sub.bx to i16
  %px.add.top = add i16 %px.trunc, %shifted_from_top
  %px.sub.by = sub i16 %px.add.top, %b_y
  ; buf[x] = (px + 1) >> 1
  %px.ext = zext i16 %px.sub.by to i32
  %px.plus1 = add nuw nsw i32 %px.ext, 1
  %result = lshr i32 %px.plus1, 1
  %result.trunc = trunc i32 %result to i8
  store i8 %result.trunc, ptr %buf.ptr, align 1
  ; tmp[x] = b_y
  store i16 %b_y, ptr %tmp.ptr, align 2
  ; x++
  %x.next = add nuw i64 %x, 1
  %cmp.x = icmp eq i64 %x.next, %width
  br i1 %cmp.x, label %outer.latch, label %inner

outer.latch:
  %buf.next = getelementptr inbounds i8, ptr %buf.y, i64 %stride
  %y.next = add nuw i64 %y, 1
  %cmp.y = icmp eq i64 %y.next, %height
  br i1 %cmp.y, label %exit, label %outer.header

exit:
  ret void
}

; WITHOUT-LABEL: define void @nested_loop_inner_vectorized(
;
; Verify the inner loop is vectorized.
; Verify there is only one vectorized copy of the inner loop.
; WITHOUT: {{^vector\.body.*:}}
; WITHOUT: add <{{[0-9]+}} x i8>
; WITHOUT: store <{{[0-9]+}} x i8>
;
; WITHOUT-NOT: {{^vector\.body.*:}}
;
; WITHOUT: ret void

; WITH-LABEL: define void @nested_loop_inner_vectorized(
; WITH: {{^vector\.body.*:}}
; WITH-NOT: {{^vector\.body.*:}}
; WITH: ret void

; WITHOUT-LABEL: define void @ass_shift_bitmap_c(
; WITHOUT: vector.body:
; WITHOUT: ret void

; WITH-LABEL: define void @ass_shift_bitmap_c(
; WITH: {{^vector\.body.*:}}
; WITH-NOT: {{^vector\.body.*:}}
; WITH: ret void

; RUN: opt -passes='print<access-info>' -disable-output %s 2>&1 | FileCheck %s

; Reverse i32 loop over 4 elements whose access range exactly fills the
; dereferenceable region (deref(16), reads bytes [0, 16)).
;
; TODO: LAA should recognise that this AR fits within the deref
; region and produce tight bounds (Low: %A, High: %A + 16).
;
; Pseudocode:
;   // A, B: at least 4 i32s dereferenceable each
;   for (i64 i = 3; i >= 0; --i) {
;     i32 l = A[i];       // A[3], A[2], A[1], A[0]
;     B[i] = 0;
;     if (l == 0) break;
;   }

define void @reverse_reaches_base(ptr dereferenceable(16) %A, ptr dereferenceable(16) %B) {
; CHECK-LABEL: 'reverse_reaches_base'
; CHECK:      Group GRP0:
; CHECK-NEXT:   (Low: (-4 + inttoptr (i64 -1 to ptr))<nsw> High: (16 + %B)<nuw>)
; CHECK-NEXT:     Member: {(12 + %B)<nuw>,+,-4}<nw><%loop>
; CHECK:      Group GRP1:
; CHECK-NEXT:   (Low: (-4 + inttoptr (i64 -1 to ptr))<nsw> High: (16 + %A)<nuw>)
; CHECK-NEXT:     Member: {(12 + %A)<nuw>,+,-4}<nw><%loop>
entry:
  br label %loop

loop:
  %iv = phi i64 [ 3, %entry ], [ %iv.dec, %latch ]
  %gep.A = getelementptr inbounds i32, ptr %A, i64 %iv
  %gep.B = getelementptr inbounds i32, ptr %B, i64 %iv
  %l = load i32, ptr %gep.A, align 4
  store i32 0, ptr %gep.B, align 4
  %uncntable = icmp eq i32 %l, 0
  br i1 %uncntable, label %exit.early, label %latch

latch:
  %iv.dec = add nsw i64 %iv, -1
  %ec = icmp eq i64 %iv, 0
  br i1 %ec, label %exit.done, label %loop

exit.early:
  ret void

exit.done:
  ret void
}

; Reverse i32 loop whose top read spills one byte past the deref end.
; The IR is UB by construction: top i32 read at byte 13 covers [13, 17),
; but deref(16) only guarantees [0, 16).
;
; TODO: LAA should reject this AR (top access exits the deref region)
; and fall back to the wide low bound.
;
; Pseudocode:
;   for (i64 i = 13; i > 1; i -= 4) {
;     i32 l = *(i32*)((char*)A + i);   // reads [i, i+4)
;     *(i32*)((char*)B + i) = 0;
;     if (l == 0) break;
;   }

define void @reverse_top_spills(ptr dereferenceable(16) %A, ptr dereferenceable(16) %B) {
; CHECK-LABEL: 'reverse_top_spills'
; CHECK:      Group GRP0:
; CHECK-NEXT:   (Low: (5 + %B)<nuw> High: (17 + %B))
; CHECK-NEXT:     Member: {(13 + %B)<nuw>,+,-4}<nw><%loop2>
; CHECK:      Group GRP1:
; CHECK-NEXT:   (Low: (5 + %A)<nuw> High: (17 + %A))
; CHECK-NEXT:     Member: {(13 + %A)<nuw>,+,-4}<nw><%loop2>
entry:
  br label %loop2

loop2:
  %iv = phi i64 [ 13, %entry ], [ %iv.dec, %latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %gep.B = getelementptr inbounds i8, ptr %B, i64 %iv
  %l = load i32, ptr %gep.A, align 1
  store i32 0, ptr %gep.B, align 1
  %uncntable = icmp eq i32 %l, 0
  br i1 %uncntable, label %exit.early, label %latch

latch:
  %iv.dec = add nsw i64 %iv, -4
  %ec = icmp eq i64 %iv, 5
  br i1 %ec, label %exit.done, label %loop2

exit.early:
  ret void

exit.done:
  ret void
}


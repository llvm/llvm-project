; RUN: opt -S -disable-output "-passes=print<scalar-evolution><use-context>,print<scalar-evolution>" %s 2>&1 | FileCheck %s --check-prefix=CTX-THEN-NOCTX
; RUN: opt -S -disable-output "-passes=print<scalar-evolution>,print<scalar-evolution><use-context>" %s 2>&1 | FileCheck %s --check-prefix=NOCTX-THEN-CTX
; RUN: opt -S -disable-output "-passes=print<scalar-evolution>,print<scalar-evolution>" %s 2>&1 | FileCheck %s --check-prefix=NOCTX-BOTH

define void @f(ptr %base, i32 %n) {
  %ext = zext i32 %n to i64
  %gep = getelementptr inbounds i8, ptr %base, i64 %ext
  ret void
}

; use-context first: shows (u nuw). Without context second: strips it.
; CTX-THEN-NOCTX-LABEL: 'f'
; CTX-THEN-NOCTX:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; CTX-THEN-NOCTX-NEXT:    -->  ((zext i32 %n to i64) + %base)(u nuw) U:
; CTX-THEN-NOCTX:       'f'
; CTX-THEN-NOCTX:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; CTX-THEN-NOCTX-NEXT:    -->  ((zext i32 %n to i64) + %base) U:

; Without context first, use-context second: use-specific flags are still
; available from cache because they are always computed.
; NOCTX-THEN-CTX-LABEL: 'f'
; NOCTX-THEN-CTX:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; NOCTX-THEN-CTX-NEXT:    -->  ((zext i32 %n to i64) + %base) U:
; NOCTX-THEN-CTX:       'f'
; NOCTX-THEN-CTX:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; NOCTX-THEN-CTX-NEXT:    -->  ((zext i32 %n to i64) + %base)(u nuw) U:

; Without context both times: no use-specific flags shown.
; NOCTX-BOTH-LABEL: 'f'
; NOCTX-BOTH:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; NOCTX-BOTH-NEXT:    -->  ((zext i32 %n to i64) + %base) U:
; NOCTX-BOTH:       'f'
; NOCTX-BOTH:         %gep = getelementptr inbounds i8, ptr %base, i64 %ext
; NOCTX-BOTH-NEXT:    -->  ((zext i32 %n to i64) + %base) U:

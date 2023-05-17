; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

%Opaque_type = type opaque
%S2i = type <{ i64, i64 }>
%D2i = type <{ i64, i64 }>
%Di = type <{ i32 }>
%Si = type <{ i32 }>

define void @B(ptr sret(%Opaque_type) %a, ptr %b, ptr %xp, ptr %yp) {
  %x = load i32, ptr %xp
  %y = load i32, ptr %yp
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret void
}

define void @C(ptr sret(%Opaque_type) %a, ptr %b, ptr %xp, ptr %yp) {
  %x = load i32, ptr %xp
  %y = load i32, ptr %yp
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret void
}

define void @A(ptr sret(%Opaque_type) %a, ptr %b, ptr %xp, ptr %yp) {
  %x = load i32, ptr %xp
  %y = load i32, ptr %yp
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret void
}

; Make sure we transfer the parameter attributes to the call site.
; CHECK-LABEL: define void @C(ptr sret
; CHECK:  tail call void @A(ptr sret(%Opaque_type) %0, ptr %1, ptr %2, ptr %3)
; CHECK:  ret void


; Make sure we transfer the parameter attributes to the call site.
; CHECK-LABEL: define void @B(ptr sret
; CHECK:  tail call void @A(ptr sret(%Opaque_type) %0, ptr %1, ptr %2, ptr %3)
; CHECK:  ret void


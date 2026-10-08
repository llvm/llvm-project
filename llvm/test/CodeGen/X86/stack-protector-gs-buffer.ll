; Check MSVC's /GS (Buffer Security Check) heuristic, selected by the
; "stack-protector-gs-buffer" function attribute.
;
; Under this heuristic the only allocas that require a protector are calls to
; alloca and the ones the frontend marked with "stack-protector" metadata;
; nothing is inferred from the alloca's type.
;
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

declare void @use(ptr)
declare void @make(ptr sret([64 x i8]))
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

;; --- Unmarked allocas --------------------------------------------------------

; An obvious buffer is not protected unless the frontend says so, because the
; IR type is not what the heuristic is defined in terms of.
; CHECK-LABEL: unmarked_array:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @unmarked_array() #0 {
  %a = alloca [64 x i8]
  call void @use(ptr %a)
  ret void
}

; Unlike sspstrong, merely taking a local's address does not protect a function.
; CHECK-LABEL: address_taken:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @address_taken() #0 {
  %a = alloca i32
  call void @use(ptr %a)
  ret void
}

;; --- Marked allocas ----------------------------------------------------------

; CHECK-LABEL: marked_small:
; CHECK:         __security_cookie
define void @marked_small() #0 {
  %a = alloca [6 x i8], !stack-protector !1
  call void @use(ptr %a)
  ret void
}

; CHECK-LABEL: marked_large:
; CHECK:         __security_cookie
define void @marked_large() #0 {
  %a = alloca [64 x i8], !stack-protector !2
  call void @use(ptr %a)
  ret void
}

; The mark is what matters, not the type: an i32 carrying it is protected.
; CHECK-LABEL: marked_scalar:
; CHECK:         __security_cookie
define void @marked_scalar() #0 {
  %a = alloca i32, !stack-protector !1
  call void @use(ptr %a)
  ret void
}

; A zero mark opts out, as it does in every other mode.
; CHECK-LABEL: marked_ignore:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @marked_ignore() #0 {
  %a = alloca [64 x i8], !stack-protector !0
  call void @use(ptr %a)
  ret void
}

;; --- alloca ------------------------------------------------------------------

; A call to alloca is a GS buffer, and is visible as such in the IR, so it
; needs no mark.
; CHECK-LABEL: dynamic_alloca:
; CHECK:         __security_cookie
define void @dynamic_alloca(i64 %n) #0 {
  %a = alloca i8, i64 %n
  call void @use(ptr %a)
  ret void
}

; Even a small constant-sized alloca is a GS buffer.
; CHECK-LABEL: small_alloca:
; CHECK:         __security_cookie
define void @small_alloca() #0 {
  %a = alloca i8, i64 3
  call void @use(ptr %a)
  ret void
}

;; --- Indirect return slots ---------------------------------------------------

; MSVC gives an object it is free to relocate and that only ever receives a
; call's indirect return value a frame slot the cookie does not guard. The
; frontend cannot see that, so the second metadata operand tells the backend the
; object is trivial and the backend checks the uses.
; CHECK-LABEL: sret_only:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @sret_only() #0 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  %v = load i8, ptr %a
  store i8 %v, ptr %a
  ret void
}

; Reading and writing through the pointer, including with a memcpy, is not an
; escape, and several returns into the same slot are still just returns.
; CHECK-LABEL: sret_twice_and_memcpy:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @sret_twice_and_memcpy(ptr %p) #0 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  call void @make(ptr sret([64 x i8]) %a)
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %a, i64 64, i1 false)
  ret void
}

; As soon as the address reaches anywhere else, MSVC moves the object into the
; guarded region instead.
; CHECK-LABEL: sret_escapes:
; CHECK:         __security_cookie
define void @sret_escapes() #0 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  call void @use(ptr %a)
  ret void
}

; A derived pointer escaping counts just the same.
; CHECK-LABEL: sret_gep_escapes:
; CHECK:         __security_cookie
define void @sret_gep_escapes() #0 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  %g = getelementptr [64 x i8], ptr %a, i64 0, i64 8
  call void @use(ptr %g)
  ret void
}

; Storing the pointer itself is an escape; storing through it is not.
; CHECK-LABEL: sret_pointer_stored:
; CHECK:         __security_cookie
define void @sret_pointer_stored(ptr %p) #0 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  store ptr %a, ptr %p
  ret void
}

; Without an indirect return there is nothing to exempt.
; CHECK-LABEL: trivial_without_sret:
; CHECK:         __security_cookie
define void @trivial_without_sret() #0 {
  %a = alloca [64 x i8], !stack-protector !3
  store i8 0, ptr %a
  ret void
}

; The exemption needs the object to be one MSVC may relocate, which is what the
; second operand says. Without it the slot is protected.
; CHECK-LABEL: nontrivial_sret:
; CHECK:         __security_cookie
define void @nontrivial_sret() #0 {
  %a = alloca [64 x i8], !stack-protector !2
  call void @make(ptr sret([64 x i8]) %a)
  ret void
}

; The exemption is specific to the /GS heuristic.
; CHECK-LABEL: strong_sret:
; CHECK:         __security_cookie
define void @strong_sret() #2 {
  %a = alloca [64 x i8], !stack-protector !3
  call void @make(ptr sret([64 x i8]) %a)
  ret void
}

;; --- Exclusions --------------------------------------------------------------

; MSVC never protects a function that takes a variable argument list, even one
; holding an obvious GS buffer.
; CHECK-LABEL: variadic:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @variadic(i32 %n, ...) #0 {
  %a = alloca [64 x i8], !stack-protector !2
  call void @use(ptr %a)
  ret void
}

; sspreq overrides the heuristic entirely, so the varargs exclusion does not
; apply.
; CHECK-LABEL: sspreq_wins:
; CHECK:         __security_cookie
define void @sspreq_wins(i32 %n, ...) #1 {
  %a = alloca i32
  call void @use(ptr %a)
  ret void
}

; Without the marker, sspstrong keeps the GCC-compatible heuristic and does
; protect a small array.
; CHECK-LABEL: strong_still_protects:
; CHECK:         __security_cookie
define void @strong_still_protects() #2 {
  %a = alloca [4 x i8]
  call void @use(ptr %a)
  ret void
}

; The marker is honoured under the other heuristics too: sspstrong would not
; protect this function on its own.
; CHECK-LABEL: strong_honours_mark:
; CHECK:         __security_cookie
define void @strong_honours_mark() #2 {
  %a = alloca i32, !stack-protector !1
  store i32 0, ptr %a
  ret void
}

attributes #0 = { sspstrong uwtable "stack-protector-gs-buffer"="true" }
attributes #1 = { sspreq uwtable "stack-protector-gs-buffer"="true" }
attributes #2 = { sspstrong uwtable }

!0 = !{i32 0}
!1 = !{i32 1}
!2 = !{i32 2}
!3 = !{i32 2, i1 true}

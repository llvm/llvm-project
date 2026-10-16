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

; A marked alloca is protected even when all it does is receive a call's
; indirect return value. cl.exe leaves such a slot alone, but telling that apart
; from a buffer that is merely returned into needs more than the backend can
; see, so it is protected like any other marked alloca.
; CHECK-LABEL: sret_slot:
; CHECK:         __security_cookie
define void @sret_slot() #0 {
  %a = alloca [64 x i8], !stack-protector !2
  call void @make(ptr sret([64 x i8]) %a)
  ret void
}

;; --- Varargs -----------------------------------------------------------------

; A function that takes a variable argument list is protected like any other;
; cl.exe does the same.
; CHECK-LABEL: variadic:
; CHECK:         __security_cookie
define void @variadic(i32 %n, ...) #0 {
  %a = alloca [64 x i8], !stack-protector !2
  call void @use(ptr %a)
  ret void
}

; Taking a variable argument list is not on its own a reason to protect, so an
; unmarked alloca in one still does not require a protector.
; CHECK-LABEL: variadic_unmarked:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @variadic_unmarked(i32 %n, ...) #0 {
  %a = alloca [64 x i8]
  call void @use(ptr %a)
  ret void
}

;; --- Other modes -------------------------------------------------------------

; sspreq overrides the heuristic entirely.
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

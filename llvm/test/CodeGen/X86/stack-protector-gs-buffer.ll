; Check MSVC's /GS (Buffer Security Check) heuristic, selected by the
; "stack-protector-gs-buffer" function attribute.
;
; A "GS buffer" is an array larger than 4 bytes with more than two elements and
; a non-pointer element type, a pointer-free aggregate larger than 8 bytes, an
; alloca of any size, or any aggregate containing one of those.
;
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

declare void @use(ptr)

;; --- Arrays that are not GS buffers ------------------------------------------

; Exactly 4 bytes, so not larger than 4.
; CHECK-LABEL: array_4_bytes:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @array_4_bytes() #0 {
  %a = alloca [4 x i8]
  call void @use(ptr %a)
  ret void
}

; 8 bytes, but only two elements.
; CHECK-LABEL: array_2_elements:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @array_2_elements() #0 {
  %a = alloca [2 x i32]
  call void @use(ptr %a)
  ret void
}

; Large and many elements, but the element type is a pointer.
; CHECK-LABEL: array_of_pointers:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @array_of_pointers() #0 {
  %a = alloca [8 x ptr]
  call void @use(ptr %a)
  ret void
}

;; --- Arrays that are GS buffers ----------------------------------------------

; CHECK-LABEL: array_8_bytes:
; CHECK:         __security_cookie
define void @array_8_bytes() #0 {
  %a = alloca [8 x i8]
  call void @use(ptr %a)
  ret void
}

; 6 bytes and three elements: over both thresholds.
; CHECK-LABEL: array_3_shorts:
; CHECK:         __security_cookie
define void @array_3_shorts() #0 {
  %a = alloca [3 x i16]
  call void @use(ptr %a)
  ret void
}

; Two elements, so the array itself is not a GS buffer, but each element is.
; CHECK-LABEL: array_2_buffers:
; CHECK:         __security_cookie
define void @array_2_buffers() #0 {
  %a = alloca [2 x [8 x i8]]
  call void @use(ptr %a)
  ret void
}

;; --- Aggregates --------------------------------------------------------------

; 12 bytes with no pointers.
; CHECK-LABEL: struct_pointer_free:
; CHECK:         __security_cookie
define void @struct_pointer_free() #0 {
  %a = alloca { i32, i32, i32 }
  call void @use(ptr %a)
  ret void
}

; 16 bytes, but it holds a pointer, so it is not itself a GS buffer.
; CHECK-LABEL: struct_with_pointer:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @struct_with_pointer() #0 {
  %a = alloca { ptr, i32, i32 }
  call void @use(ptr %a)
  ret void
}

; Exactly 8 bytes, so not larger than 8.
; CHECK-LABEL: struct_8_bytes:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @struct_8_bytes() #0 {
  %a = alloca { i32, i32 }
  call void @use(ptr %a)
  ret void
}

; Holds a pointer, but also contains a GS buffer.
; CHECK-LABEL: struct_containing_buffer:
; CHECK:         __security_cookie
define void @struct_containing_buffer() #0 {
  %a = alloca { ptr, [8 x i8] }
  call void @use(ptr %a)
  ret void
}

; The GS buffer is two aggregates down.
; CHECK-LABEL: struct_nested_buffer:
; CHECK:         __security_cookie
define void @struct_nested_buffer() #0 {
  %a = alloca { ptr, { i32, [8 x i8] } }
  call void @use(ptr %a)
  ret void
}

;; --- alloca ------------------------------------------------------------------

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

;; --- Exclusions --------------------------------------------------------------

; Unlike sspstrong, merely taking a local's address does not protect a function.
; CHECK-LABEL: address_taken:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @address_taken() #0 {
  %a = alloca i32
  call void @use(ptr %a)
  ret void
}

; MSVC never protects a function that takes a variable argument list, even one
; holding an obvious GS buffer.
; CHECK-LABEL: variadic:
; CHECK-NOT:     __security_cookie
; CHECK:       .seh_endproc
define void @variadic(i32 %n, ...) #0 {
  %a = alloca [64 x i8]
  call void @use(ptr %a)
  ret void
}

; sspreq overrides the heuristic entirely, so the varargs exclusion and the
; GS-buffer rules do not apply.
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

attributes #0 = { sspstrong uwtable "stack-protector-gs-buffer"="true" }
attributes #1 = { sspreq uwtable "stack-protector-gs-buffer"="true" }
attributes #2 = { sspstrong uwtable }

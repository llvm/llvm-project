; RUN: llc -mtriple aarch64-arm-linux-gnu --enable-machine-outliner \
; RUN: -verify-machineinstrs %s -o - | FileCheck --check-prefixes CHECK,V8A %s
; RUN-V83A: llc -mtriple aarch64-arm-none-eabi -enable-machine-outliner \
; RUN-V83A: -verify-machineinstrs -mattr=+v8.3a %s -o - > %t
; RUN-V83A: FileCheck --check-prefixes CHECK,V83A < %t %s

declare i32 @thunk_called_fn(i32, i32, i32, i32)

define i32 @a() #0 {
; CHECK-LABEL:  a:                                      // @a
; CHECK:        // %bb.0:                               // %entry
; V8A-NEXT:         hint #25
; V83A-NEXT:        paciasp
; V8A:              hint #29
; V83A:             autiasp
; CHECK-NEXT:       ret
entry:
  %call = tail call i32 @thunk_called_fn(i32 1, i32 2, i32 3, i32 4)
  %cx = add i32 %call, 8
  ret i32 %cx
}

define i32 @b() #0 {
; CHECK-LABEL:  b:                                      // @b
; CHECK:        // %bb.0:                               // %entry
; V8A-NEXT:         hint #25
; V83A-NEXT:        paciasp
; CHECK-NEXT:       .cfi_negate_ra_state
; V8A:              hint #29
; V83A:             autiasp
; CHECK-NEXT:       ret
entry:
  %call = tail call i32 @thunk_called_fn(i32 1, i32 2, i32 3, i32 4)
  %cx = add i32 %call, 88
  ret i32 %cx
}

define hidden i32 @c(i32 (i32, i32, i32, i32)* %fptr) #0 {
; CHECK-LABEL:  c:                                      // @c
; CHECK:        // %bb.0:                               // %entry
; V8A-NEXT:         hint #25
; V83A-NEXT:        paciasp
; CHECK-NEXT:       .cfi_negate_ra_state
; V8A:              hint #29
; V83A:             autiasp
; CHECK-NEXT:       ret
entry:
  %call = tail call i32 %fptr(i32 1, i32 2, i32 3, i32 4)
  %add = add nsw i32 %call, 8
  ret i32 %add
}

define hidden i32 @d(i32 (i32, i32, i32, i32)* %fptr) #0 {
; CHECK-LABEL:  d:                                      // @d
; CHECK:        // %bb.0:                               // %entry
; V8A-NEXT:         hint #25
; V83A-NEXT:        paciasp
; CHECK-NEXT:       .cfi_negate_ra_state
; V8A:              hint #29
; V83A:             autiasp
; CHECK-NEXT:       ret
entry:
  %call = tail call i32 %fptr(i32 1, i32 2, i32 3, i32 4)
  %add = add nsw i32 %call, 88
  ret i32 %add
}

attributes #0 = { "sign-return-address"="non-leaf" }

; CHECK-NOT:        [[OUTLINED_FUNCTION_{{.*}}]]
; CHECK-NOT:         .cfi_b_key_frame
; CHECK-NOT:         paci{{[a,b]}}sp
; CHECK-NOT:         hint #2{{[5,7]}}
; CHECK-NOT:         .cfi_negate_ra_state
; CHECK-NOT:         auti{{[a,b]}}sp
; CHECK-NOT:         hint #{{[29,31]}}

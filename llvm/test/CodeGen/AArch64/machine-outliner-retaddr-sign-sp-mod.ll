; RUN: llc -mtriple aarch64 -enable-machine-outliner \
; RUN:  -verify-machineinstrs %s -o - | FileCheck %s

@v = common dso_local global ptr null, align 8

define dso_local void @foo(i32 %x) #0 {
; CHECK-LABEL:  foo:                                    // @foo
; CHECK-NOT:        bl OUTLINED_FUNCTION_
; CHECK:        // %bb.0:                               // %entry
; CHECK-NEXT:       paciasp
; CHECK-NEXT:       stp x29, x30, [sp, #-16]!
; CHECK-NEXT:       // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:       ubfiz x8, x0, #2, #32
; CHECK-NEXT:       mov x9, sp
; CHECK-NEXT:       mov x29, sp
; CHECK-NEXT:       add x8, x8, #15
; CHECK-NEXT:       and x8, x8, #0x7fffffff0
; CHECK-NEXT:       sub x8, x9, x8
; CHECK-NEXT:       mov sp, x8
; CHECK-NEXT:       adrp x9, v
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       mov sp, x29
; CHECK-NEXT:       ldp x29, x30, [sp], #16
; CHECK-NEXT:       retaa
entry:
  %0 = zext i32 %x to i64
  %vla = alloca i32, i64 %0, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

define dso_local void @bar(i32 %x) #0 {
; CHECK-LABEL:  bar:                                    // @bar
; CHECK-NOT:        bl OUTLINED_FUNCTION_
; CHECK:        // %bb.0:                               // %entry
; CHECK-NEXT:       paciasp
; CHECK-NEXT:       stp x29, x30, [sp, #-16]!
; CHECK-NEXT:       // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:       ubfiz x8, x0, #2, #32
; CHECK-NEXT:       mov x9, sp
; CHECK-NEXT:       mov x29, sp
; CHECK-NEXT:       add x8, x8, #15
; CHECK-NEXT:       and x8, x8, #0x7fffffff0
; CHECK-NEXT:       sub x8, x9, x8
; CHECK-NEXT:       mov sp, x8
; CHECK-NEXT:       adrp x9, v
; CHECK-NEXT:       str xzr, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       str x8, [x9, :lo12:v]
; CHECK-NEXT:       mov sp, x29
; CHECK-NEXT:       ldp x29, x30, [sp], #16
; CHECK-NEXT:       retaa
entry:
  %0 = zext i32 %x to i64
  %vla = alloca i32, i64 %0, align 4
  store volatile ptr null, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

attributes #0 = { nounwind "target-features"="+v8.3a" "frame-pointer"="all" "sign-return-address"="all" "sign-return-address-key"="a_key" }

; CHECK-NOT: OUTLINED_FUNCTION_

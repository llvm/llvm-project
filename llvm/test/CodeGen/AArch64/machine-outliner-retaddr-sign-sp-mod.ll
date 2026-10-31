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

; Non-tail call (BL) must not be treated as an illegal SP modification.
define void @leaf() noinline #1 {
; CHECK-LABEL:  leaf:
  ret void
}

define dso_local void @nontail_call_a(ptr %p) minsize #1 {
; CHECK-LABEL: nontail_call_a:
; CHECK: bl OUTLINED_FUNCTION_1
entry:
  call void @leaf()
  call void @leaf()
  call void @leaf()
  store volatile i32 1, ptr %p
  store volatile i32 2, ptr %p
  store volatile i32 3, ptr %p
  ret void
}

define dso_local void @nontail_call_b(ptr %p) minsize #1 {
; CHECK-LABEL: nontail_call_b:
; CHECK: bl OUTLINED_FUNCTION_1
entry:
  call void @leaf()
  call void @leaf()
  call void @leaf()
  store volatile i32 1, ptr %p
  store volatile i32 2, ptr %p
  store volatile i32 3, ptr %p
  ret void
}

; Indirect call (BLR) with PAC + BTI must not be treated as an illegal SP modification.
%struct.obj = type { ptr, [8 x i32] }

define dso_local i32 @indirect_call_a(ptr noundef %0) minsize #2 {
; CHECK-LABEL: indirect_call_a:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  %p0 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 0
  %p1 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 1
  %v1 = load i32, ptr %p1, align 4
  %a1 = add nsw i32 %v1, 3
  %x1 = xor i32 %a1, 85
  store i32 %x1, ptr %p0, align 8
  %p2 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 2
  %p3 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 3
  %v3 = load i32, ptr %p3, align 4
  %m3 = mul nsw i32 %v3, 5
  %a3 = add nsw i32 %m3, 17
  store i32 %a3, ptr %p2, align 8
  %vtbl_ptr = load ptr, ptr %0, align 8
  %fn_ptr_slot = getelementptr inbounds ptr, ptr %vtbl_ptr, i64 3
  %fn = load ptr, ptr %fn_ptr_slot, align 8
  %call = tail call i32 %fn(ptr noundef nonnull %0, i32 noundef %x1, i32 noundef %a3, i32 noundef 0)
  %res = add nsw i32 %call, 1
  ret i32 %res
}

define dso_local i32 @indirect_call_b(ptr noundef %0) minsize #2 {
; CHECK-LABEL: indirect_call_b:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  %p0 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 0
  %p1 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 1
  %v1 = load i32, ptr %p1, align 4
  %a1 = add nsw i32 %v1, 3
  %x1 = xor i32 %a1, 85
  store i32 %x1, ptr %p0, align 8
  %p2 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 2
  %p3 = getelementptr inbounds %struct.obj, ptr %0, i64 0, i32 1, i64 3
  %v3 = load i32, ptr %p3, align 4
  %m3 = mul nsw i32 %v3, 5
  %a3 = add nsw i32 %m3, 17
  store i32 %a3, ptr %p2, align 8
  %vtbl_ptr = load ptr, ptr %0, align 8
  %fn_ptr_slot = getelementptr inbounds ptr, ptr %vtbl_ptr, i64 3
  %fn = load ptr, ptr %fn_ptr_slot, align 8
  %call = tail call i32 %fn(ptr noundef nonnull %0, i32 noundef %x1, i32 noundef %a3, i32 noundef 0)
  %res = add nsw i32 %call, 2
  ret i32 %res
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK: hint #25
; CHECK: blr x8
; CHECK: hint #29
; CHECK: ret

; CHECK-LABEL: OUTLINED_FUNCTION_1:
; CHECK: hint #25
; CHECK: bl leaf
; CHECK: hint #29
; CHECK: ret

attributes #0 = { nounwind "target-features"="+v8.3a" "frame-pointer"="all" "sign-return-address"="all" "sign-return-address-key"="a_key" }
attributes #1 = { "sign-return-address"="non-leaf" }
attributes #2 = { "sign-return-address"="non-leaf" "branch-target-enforcement" }

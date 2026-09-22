define i32 @add_sub_shared_const_i32(i32 %a, i32 %b) {
; CHECK-LABEL: add_sub_shared_const_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    mov w8, #4097 // =0x1001
; CHECK-NEXT:    sub w1, w1, w8
; CHECK-NEXT:    add w0, w0, w8
; CHECK-NEXT:    bl use32
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret
  %add = add i32 %a, 4097
  %sub = add i32 %b, -4097
  %r = call i32 @use32(i32 %add, i32 %sub)
  ret i32 %r
}
declare i32 @use32(i32, i32)

define i64 @add_sub_shared_const_i64(i64 %a, i64 %b) {
; CHECK-LABEL: add_sub_shared_const_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    mov x8, #4097 // =0x1001
; CHECK-NEXT:    mov w9, #4097 // =0x1001
; CHECK-NEXT:    sub x1, x1, x8
; CHECK-NEXT:    add x0, x0, x9
; CHECK-NEXT:    bl use64
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret
  %add = add i64 %a, 4097
  %sub = add i64 %b, -4097
  %r = call i64 @use64(i64 %add, i64 %sub)
  ret i64 %r
}
declare i64 @use64(i64, i64)

define i64 @add_neg_tied_const_does_not_break_madd(i64 %a) {
; CHECK-LABEL: add_neg_tied_const_does_not_break_madd:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #37 // =0x25
; CHECK-NEXT:    mov x9, #-32888 // =0xffffffffffff7f88
; CHECK-NEXT:    movk x9, #65518, lsl #16
; CHECK-NEXT:    madd x0, x0, x8, x9
; CHECK-NEXT:    ret
  %tmp0 = add i64 %a, -31000
  %tmp1 = mul i64 %tmp0, 37
  ret i64 %tmp1
}

define i32 @add_self_negating_const(i32 %a) {
; CHECK-LABEL: add_self_negating_const:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #-2147483648 // =0x80000000
; CHECK-NEXT:    add w0, w0, w8
; CHECK-NEXT:    ret
  %add = add i32 %a, -2147483648
  ret i32 %add
}

; RUN: llc -relocation-model=pic < %s | FileCheck %s
; PR5723
target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { [1 x i64] }
%link = type { ptr }
%test = type { i32, %link }

@data = global [2 x i64] zeroinitializer, align 64 ; <ptr> [#uses=1]
@ptr = linkonce thread_local global [1 x i64] [i64 ptrtoint (ptr @data to i64)], align 64 ; <ptr> [#uses=1]
@link_ptr = linkonce thread_local global [1 x i64] zeroinitializer, align 64 ; <ptr> [#uses=1]
@_dm_my_pe = external global [1 x i64], align 64  ; <ptr> [#uses=0]
@_dm_pes_in_prog = external global [1 x i64], align 64 ; <ptr> [#uses=0]
@_dm_npes_div_mult = external global [1 x i64], align 64 ; <ptr> [#uses=0]
@_dm_npes_div_shift = external global [1 x i64], align 64 ; <ptr> [#uses=0]
@_dm_pe_addr_loc = external global [1 x i64], align 64 ; <ptr> [#uses=0]
@_dm_offset_addr_mask = external global [1 x i64], align 64 ; <ptr> [#uses=0]

define void @leaf() nounwind {
; CHECK-LABEL: leaf:
; CHECK-NOT: -8(%rsp)
; CHECK: leaq link_ptr@TLSGD
; CHECK: callq __tls_get_addr@PLT
"file foo2.c, line 14, bb1":
  %p = alloca ptr, align 8                     ; <ptr> [#uses=4]
  br label %"file foo2.c, line 14, bb2"

"file foo2.c, line 14, bb2":                      ; preds = %"file foo2.c, line 14, bb1"
  br label %"@CFE_debug_label_0"

"@CFE_debug_label_0":                             ; preds = %"file foo2.c, line 14, bb2"
  %r = load ptr, ptr @ptr, align 8 ; <ptr> [#uses=1]
  store ptr %r, ptr %p, align 8
  br label %"@CFE_debug_label_2"

"@CFE_debug_label_2":                             ; preds = %"@CFE_debug_label_0"
  %r1 = load ptr, ptr @link_ptr, align 8 ; <ptr> [#uses=1]
  %r2 = load ptr, ptr %p, align 8                  ; <ptr> [#uses=1]
  %r3 = ptrtoint ptr %r2 to i64                ; <i64> [#uses=1]
  %r4 = inttoptr i64 %r3 to ptr               ; <ptr> [#uses=1]
  %r5 = getelementptr ptr, ptr %r4, i64 1          ; <ptr> [#uses=1]
  store ptr %r1, ptr %r5, align 8
  br label %"@CFE_debug_label_3"

"@CFE_debug_label_3":                             ; preds = %"@CFE_debug_label_2"
  %r6 = load ptr, ptr %p, align 8                  ; <ptr> [#uses=1]
  %r7 = ptrtoint ptr %r6 to i64                ; <i64> [#uses=1]
  %r8 = inttoptr i64 %r7 to ptr                ; <ptr> [#uses=1]
  %r9 = getelementptr %link, ptr %r8, i64 1           ; <ptr> [#uses=1]
  store ptr %r9, ptr @link_ptr, align 8
  br label %"@CFE_debug_label_4"

"@CFE_debug_label_4":                             ; preds = %"@CFE_debug_label_3"
  %r10 = load ptr, ptr %p, align 8                 ; <ptr> [#uses=1]
  %r11 = ptrtoint ptr %r10 to i64              ; <i64> [#uses=1]
  %r12 = inttoptr i64 %r11 to ptr                ; <ptr> [#uses=1]
  store i32 1, ptr %r12, align 4
  br label %"@CFE_debug_label_5"

"@CFE_debug_label_5":                             ; preds = %"@CFE_debug_label_4"
  ret void
}

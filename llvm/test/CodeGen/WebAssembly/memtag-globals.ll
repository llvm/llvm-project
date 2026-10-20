; RUN: llc < %s -stop-after=WebAssembly-globals-tagging | FileCheck %s --check-prefix=IR
; RUN: llc < %s -asm-verbose=false | FileCheck %s --check-prefix=ASM

target triple = "wasm32-unknown-unknown"

; Globals smaller than a tag granule are padded to 16 bytes and 16-byte aligned.
; IR: @tagged = {{.*}}global { i32, [12 x i8] } { i32 42, [12 x i8] zeroinitializer }, sanitize_memtag, align 16
@tagged = global i32 42, sanitize_memtag, align 4

; Untagged globals are left alone.
; IR: @plain = {{.*}}global i32 7
@plain = global i32 7, align 4

; Globals that are already granule-sized only get their alignment bumped.
; IR: @arr = {{.*}}global [4 x i32] zeroinitializer, sanitize_memtag, align 16
@arr = global [4 x i32] zeroinitializer, sanitize_memtag, align 4

; Constant globals are not tagged; their metadata is dropped.
; IR: @rodata = constant i32 5, align 4
@rodata = constant i32 5, sanitize_memtag, align 4

; TLS globals are not tagged; their metadata is dropped.
; IR: @tls = thread_local global i32 9, align 4
@tls = thread_local global i32 9, sanitize_memtag, align 4

; @pointed_to is used by a global initializer, which holds a link-time constant,
; so it is left untagged (tag 0), and its metadata is dropped.
; IR: @pointed_to = global i32 11, align 4
@pointed_to = global i32 11, sanitize_memtag, align 4

; @holder is not referenced by any initializer or alias, so it is tagged.
; IR: @holder = {{.*}}global { ptr, [12 x i8] } { ptr @pointed_to, [12 x i8] zeroinitializer }, sanitize_memtag, align 16
@holder = global ptr @pointed_to, sanitize_memtag, align 8

; A single memtag.load is hoisted to the function entry for each tagged global,
; and every use of the global goes through it.
; IR-LABEL: define i32 @use(
; IR:         %[[P:.*]] = call ptr @llvm.wasm.memtag.load(i32 0, ptr @tagged)
; IR-NEXT:    %[[A:.*]] = load volatile i32, ptr %[[P]], align 4
; IR-NEXT:    store volatile i32 %x, ptr %[[P]], align 4
define i32 @use(i32 %x) {
  %a = load volatile i32, ptr @tagged, align 4
  store volatile i32 %x, ptr @tagged, align 4
  ret i32 %a
}

; Constant expressions containing a tagged global are materialized as
; instructions over the tagged pointer.
; IR-LABEL: define ptr @gep(
; IR:         %[[P:.*]] = call ptr @llvm.wasm.memtag.load(i32 0, ptr @arr)
; IR-NEXT:    %[[G:.*]] = getelementptr inbounds [4 x i32], ptr %[[P]], i64 0, i64 2
; IR-NEXT:    ret ptr %[[G]]
define ptr @gep() {
  ret ptr getelementptr inbounds ([4 x i32], ptr @arr, i64 0, i64 2)
}

; Untagged globals keep using their raw address.
; IR-LABEL: define i32 @use_untagged(
; IR-NOT:     memtag.load
; IR:         %[[A:.*]] = load i32, ptr @plain
; IR-NEXT:    ret i32 %[[A]]
define i32 @use_untagged() {
  %a = load i32, ptr @plain, align 4
  ret i32 %a
}

; The data symbol carries the memtag symbol attribute and the padded size.
; ASM:      memtag.load 0
; ASM:      .memtag tagged
; ASM:      .size tagged, 16
; ASM:      .memtag arr
; ASM:      .size arr, 16
; ASM-NOT:  .memtag plain
; ASM-NOT:  .memtag rodata
; ASM-NOT:  .memtag pointed_to

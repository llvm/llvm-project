; RUN: llc < %s --mtriple=wasm32-wasip3 -asm-verbose=false | FileCheck --check-prefix=LIBCALL %s
; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false | FileCheck --check-prefix=GLOBAL %s

declare void @force_sp_save()
define void @use_stack() #0 {
  %1 = alloca i32, align 4
  %2 = alloca ptr, align 4
  store ptr %1, ptr %2, align 4
  call void @force_sp_save()
  ret void
}

; LIBCALL-LABEL: use_stack:
; LIBCALL: call __wasm_get_stack_pointer
; LIBCALL: call __wasm_set_stack_pointer
; LIBCALL-NOT: global.get __stack_pointer
; LIBCALL-NOT: global.set __stack_pointer

; GLOBAL-LABEL: use_stack:
; GLOBAL: global.get __stack_pointer
; GLOBAL: global.set __stack_pointer
; GLOBAL-NOT: call __wasm_get_stack_pointer
; GLOBAL-NOT: call __wasm_set_stack_pointer


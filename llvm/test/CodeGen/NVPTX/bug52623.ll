; RUN: llc < %s -march=nvptx -verify-machineinstrs
; RUN: %if ptxas %{ llc < %s -march=nvptx | %ptxas-verify %}

; Check that llc will not crash even when first MBB doesn't contain
; any instruction.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64"

%printf_args.0.8 = type { i8* }

define internal i32 @__kmpc_get_hardware_thread_id_in_block(i1 %0) {
  %2 = alloca %printf_args.0.8, i32 0, align 8
  %3 = bitcast %printf_args.0.8* %2 to i8*
  br i1 true, label %._crit_edge1, label %._crit_edge

._crit_edge:                                      ; preds = %1, %._crit_edge
  %4 = call i32 null(i8* null, i8* %3)
  br i1 %0, label %._crit_edge, label %._crit_edge1

._crit_edge1:                                     ; preds = %._crit_edge, %1
  ret i32 0

; uselistorder directives
  uselistorder label %._crit_edge, { 1, 0 }
}

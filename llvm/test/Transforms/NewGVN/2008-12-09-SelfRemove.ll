; RUN: opt < %s -passes=newgvn -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.anon = type { ptr, i32 }
	%struct.d_print_info = type { i32, ptr, i32, i32, ptr, ptr, i32 }
	%struct.d_print_mod = type { ptr, ptr, i32, ptr }
	%struct.d_print_template = type { ptr, ptr }
	%struct.demangle_component = type { i32, { %struct.anon } }

define void @d_print_mod_list(ptr %dpi, ptr %mods, i32 %suffix) nounwind {
entry:
	%0 = getelementptr %struct.d_print_info, ptr %dpi, i32 0, i32 1		; <ptr> [#uses=1]
	br i1 false, label %return, label %bb

bb:		; preds = %entry
	%1 = load ptr, ptr %0, align 4		; <ptr> [#uses=0]
	%2 = getelementptr %struct.d_print_info, ptr %dpi, i32 0, i32 1		; <ptr> [#uses=0]
	br label %bb21

bb21:		; preds = %bb21, %bb
	br label %bb21

return:		; preds = %entry
	ret void
}

; CHECK: define void @d_print_mod_list(ptr %dpi, ptr %mods, i32 %suffix) #0 {
; CHECK: entry:
; CHECK:   %0 = getelementptr %struct.d_print_info, ptr %dpi, i32 0, i32 1
; CHECK:   br i1 false, label %return, label %bb
; CHECK: bb:
; CHECK:   br label %bb21
; CHECK: bb21:
; CHECK:   br label %bb21
; CHECK: return:
; CHECK:   ret void
; CHECK: }

; RUN: opt < %s -gvn -S | grep getelementptr | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.anon = type { i8*, i32 }
	%struct.d_print_info = type { i32, i8*, i32, i32, %struct.d_print_template*, %struct.d_print_mod*, i32 }
	%struct.d_print_mod = type { %struct.d_print_mod*, %struct.demangle_component*, i32, %struct.d_print_template* }
	%struct.d_print_template = type { %struct.d_print_template*, %struct.demangle_component* }
	%struct.demangle_component = type { i32, { %struct.anon } }

define void @d_print_mod_list(%struct.d_print_info* %dpi, %struct.d_print_mod* %mods, i32 %suffix) nounwind {
entry:
	%0 = getelementptr %struct.d_print_info* %dpi, i32 0, i32 1		; <i8**> [#uses=1]
	br i1 false, label %return, label %bb

bb:		; preds = %entry
	%1 = load i8** %0, align 4		; <i8*> [#uses=0]
	%2 = getelementptr %struct.d_print_info* %dpi, i32 0, i32 1		; <i8**> [#uses=0]
	br label %bb21

bb21:		; preds = %bb21, %bb
	br label %bb21

return:		; preds = %entry
	ret void
}

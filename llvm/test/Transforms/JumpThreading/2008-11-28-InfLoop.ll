; RUN: llvm-as < %s | opt -jump-threading | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.decContext = type { i32 }
	%struct.decNumber = type { i32, i32 }

define i32 @decNumberPower(%struct.decNumber* %res, %struct.decNumber* %lhs, %struct.decNumber* %rhs, %struct.decContext* %set) nounwind {
entry:
	br i1 true, label %decDivideOp.exit, label %bb7.i

bb7.i:		; preds = %bb7.i, %entry
	br label %bb7.i

decDivideOp.exit:		; preds = %entry
	ret i32 undef
}

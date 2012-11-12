; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR4066
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.i2c_device_id = type { }
@w83l785ts_id = internal constant [0 x %struct.i2c_device_id] zeroinitializer, align 1		; <[0 x %struct.i2c_device_id]*> [#uses=1]

@__mod_i2c_device_table = alias getelementptr ([0 x %struct.i2c_device_id]* @w83l785ts_id, i32 0, i32 0)		; <%struct.i2c_device_id*> [#uses=0]

; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {alloca.*client_t}
; PR1446
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"

	%struct.clientSnapshot_t = type { i32, [32 x i8], %struct.playerState_t, i32, i32, i32, i32, i32 }
	%struct.client_t = type { i32, [1024 x i8], [64 x [1024 x i8]], i32, i32, i32, i32, i32, i32, %struct.usercmd_t, i32, i32, [1024 x i8], %struct.sharedEntity_t*, [32 x i8], [64 x i8], i32, i32, i32, i32, i32, i32, [8 x i8*], [8 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x %struct.clientSnapshot_t], i32, i32, i32, i32, i32, %struct.netchan_t, %struct.netchan_buffer_t*, %struct.netchan_buffer_t**, i32, [1025 x i32] }
	%struct.entityShared_t = type { %struct.entityState_t, i32, i32, i32, i32, i32, [3 x float], [3 x float], i32, [3 x float], [3 x float], [3 x float], [3 x float], i32 }
	%struct.entityState_t = type { i32, i32, i32, %struct.trajectory_t, %struct.trajectory_t, i32, i32, [3 x float], [3 x float], [3 x float], [3 x float], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.msg_t = type { i32, i32, i32, i8*, i32, i32, i32, i32 }
	%struct.netadr_t = type { i32, [4 x i8], [10 x i8], i16 }
	%struct.netchan_buffer_t = type { %struct.msg_t, [16384 x i8], %struct.netchan_buffer_t* }
	%struct.netchan_t = type { i32, i32, %struct.netadr_t, i32, i32, i32, i32, i32, [16384 x i8], i32, i32, i32, [16384 x i8] }
	%struct.playerState_t = type { i32, i32, i32, i32, i32, [3 x float], [3 x float], i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, [3 x float], i32, i32, [2 x i32], [2 x i32], i32, i32, i32, i32, i32, i32, [3 x float], i32, i32, i32, i32, i32, [16 x i32], [16 x i32], [16 x i32], [16 x i32], i32, i32, i32, i32, i32, i32, i32 }
	%struct.sharedEntity_t = type { %struct.entityState_t, %struct.entityShared_t }
	%struct.trajectory_t = type { i32, i32, i32, [3 x float], [3 x float] }
	%struct.usercmd_t = type { i32, [3 x i32], i32, i8, i8, i8, i8 }

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define void @SV_DirectConnect(i64 %from.0.0, i64 %from.0.1, i32 %from.1) {
entry:
	%temp = alloca %struct.client_t, align 16		; <%struct.client_t*> [#uses=1]
	%temp586 = bitcast %struct.client_t* %temp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* null, i8* %temp586, i32 121596, i32 0 )
	unreachable
}

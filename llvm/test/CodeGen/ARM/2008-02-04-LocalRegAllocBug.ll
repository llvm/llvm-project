; RUN: llc < %s -mtriple=arm-linux-gnueabi -regalloc=fast -optimize-regalloc=0
; PR1925

	%struct.encode_aux_nearestmatch = type { ptr, ptr, ptr, ptr, i32, i32 }
	%struct.encode_aux_pigeonhole = type { float, float, i32, i32, ptr, i32, ptr, ptr, ptr }
	%struct.encode_aux_threshmatch = type { ptr, ptr, i32, i32 }
	%struct.oggpack_buffer = type { i32, i32, ptr, ptr, i32 }
	%struct.static_codebook = type { i32, i32, ptr, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32 }

define i32 @vorbis_staticbook_pack(ptr %c, ptr %opb) {
entry:
	%opb_addr = alloca ptr		; <ptr> [#uses=1]
	%tmp1 = load ptr, ptr %opb_addr, align 4		; <ptr> [#uses=1]
	call void @oggpack_write( ptr %tmp1, i32 5653314, i32 24 ) nounwind 
	call void @oggpack_write( ptr null, i32 0, i32 24 ) nounwind 
	unreachable
}

declare void @oggpack_write(ptr, i32, i32)

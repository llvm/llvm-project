; RUN: llc < %s -mtriple=arm-linux-gnueabi -regalloc=fast -optimize-regalloc=0
; PR1925

	%struct.encode_aux_nearestmatch = type { i32*, i32*, i32*, i32*, i32, i32 }
	%struct.encode_aux_pigeonhole = type { float, float, i32, i32, i32*, i32, i32*, i32*, i32* }
	%struct.encode_aux_threshmatch = type { float*, i32*, i32, i32 }
	%struct.oggpack_buffer = type { i32, i32, i8*, i8*, i32 }
	%struct.static_codebook = type { i32, i32, i32*, i32, i32, i32, i32, i32, i32*, %struct.encode_aux_nearestmatch*, %struct.encode_aux_threshmatch*, %struct.encode_aux_pigeonhole*, i32 }

define i32 @vorbis_staticbook_pack(%struct.static_codebook* %c, %struct.oggpack_buffer* %opb) {
entry:
	%opb_addr = alloca %struct.oggpack_buffer*		; <%struct.oggpack_buffer**> [#uses=1]
	%tmp1 = load %struct.oggpack_buffer** %opb_addr, align 4		; <%struct.oggpack_buffer*> [#uses=1]
	call void @oggpack_write( %struct.oggpack_buffer* %tmp1, i32 5653314, i32 24 ) nounwind 
	call void @oggpack_write( %struct.oggpack_buffer* null, i32 0, i32 24 ) nounwind 
	unreachable
}

declare void @oggpack_write(%struct.oggpack_buffer*, i32, i32)

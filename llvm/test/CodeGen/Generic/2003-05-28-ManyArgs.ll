; RUN: llc < %s

;; Date:     May 28, 2003.
;; From:     test/Programs/External/SPEC/CINT2000/175.vpr.llvm.bc
;; Function: int %main(int %argc.1, sbyte** %argv.1)
;;
;; Error:    A function call with about 56 arguments causes an assertion failure
;;           in llc because the register allocator cannot find a register
;;           not used explicitly by the call instruction.
;;
;; Cause:    Regalloc was not keeping track of free registers correctly.
;;           It was counting the registers allocated to all outgoing arguments,
;;           even though most of those are copied to the stack (so those
;;           registers are not actually used by the call instruction).
;;
;; Fixed:    By rewriting selection and allocation so that selection explicitly
;;           inserts all copy operations required for passing arguments and
;;           for the return value of a call, copying to/from registers
;;           and/or to stack locations as needed.
;;
	%struct..s_annealing_sched = type { i32, float, float, float, float }
	%struct..s_chan = type { i32, float, float, float, float }
	%struct..s_det_routing_arch = type { i32, float, float, float, i32, i32, i16, i16, i16, float, float }
	%struct..s_placer_opts = type { i32, float, i32, i32, i8*, i32, i32 }
	%struct..s_router_opts = type { float, float, float, float, float, i32, i32, i32, i32 }
	%struct..s_segment_inf = type { float, i32, i16, i16, float, float, i32, float, float }
	%struct..s_switch_inf = type { i32, float, float, float, float }

define i32 @main(i32 %argc.1, i8** %argv.1) {
entry:
	%net_file = alloca [300 x i8]		; <[300 x i8]*> [#uses=1]
	%place_file = alloca [300 x i8]		; <[300 x i8]*> [#uses=1]
	%arch_file = alloca [300 x i8]		; <[300 x i8]*> [#uses=1]
	%route_file = alloca [300 x i8]		; <[300 x i8]*> [#uses=1]
	%full_stats = alloca i32		; <i32*> [#uses=1]
	%operation = alloca i32		; <i32*> [#uses=1]
	%verify_binary_search = alloca i32		; <i32*> [#uses=1]
	%show_graphics = alloca i32		; <i32*> [#uses=1]
	%annealing_sched = alloca %struct..s_annealing_sched		; <%struct..s_annealing_sched*> [#uses=5]
	%placer_opts = alloca %struct..s_placer_opts		; <%struct..s_placer_opts*> [#uses=7]
	%router_opts = alloca %struct..s_router_opts		; <%struct..s_router_opts*> [#uses=9]
	%det_routing_arch = alloca %struct..s_det_routing_arch		; <%struct..s_det_routing_arch*> [#uses=11]
	%segment_inf = alloca %struct..s_segment_inf*		; <%struct..s_segment_inf**> [#uses=1]
	%timing_inf = alloca { i32, float, float, float, float, float, float, float, float, float, float }		; <{ i32, float, float, float, float, float, float, float, float, float, float }*> [#uses=11]
	%tmp.101 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 4		; <i8**> [#uses=1]
	%tmp.105 = getelementptr [300 x i8], [300 x i8]* %net_file, i64 0, i64 0		; <i8*> [#uses=1]
	%tmp.106 = getelementptr [300 x i8], [300 x i8]* %arch_file, i64 0, i64 0		; <i8*> [#uses=1]
	%tmp.107 = getelementptr [300 x i8], [300 x i8]* %place_file, i64 0, i64 0		; <i8*> [#uses=1]
	%tmp.108 = getelementptr [300 x i8], [300 x i8]* %route_file, i64 0, i64 0		; <i8*> [#uses=1]
	%tmp.109 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 0		; <i32*> [#uses=1]
	%tmp.112 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 0		; <i32*> [#uses=1]
	%tmp.114 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 6		; <i32*> [#uses=1]
	%tmp.118 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 7		; <i32*> [#uses=1]
	%tmp.135 = load i32, i32* %operation		; <i32> [#uses=1]
	%tmp.137 = load i32, i32* %tmp.112		; <i32> [#uses=1]
	%tmp.138 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 1		; <float*> [#uses=1]
	%tmp.139 = load float, float* %tmp.138		; <float> [#uses=1]
	%tmp.140 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 2		; <i32*> [#uses=1]
	%tmp.141 = load i32, i32* %tmp.140		; <i32> [#uses=1]
	%tmp.142 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 3		; <i32*> [#uses=1]
	%tmp.143 = load i32, i32* %tmp.142		; <i32> [#uses=1]
	%tmp.145 = load i8*, i8** %tmp.101		; <i8*> [#uses=1]
	%tmp.146 = getelementptr %struct..s_placer_opts, %struct..s_placer_opts* %placer_opts, i64 0, i32 5		; <i32*> [#uses=1]
	%tmp.147 = load i32, i32* %tmp.146		; <i32> [#uses=1]
	%tmp.149 = load i32, i32* %tmp.114		; <i32> [#uses=1]
	%tmp.154 = load i32, i32* %full_stats		; <i32> [#uses=1]
	%tmp.155 = load i32, i32* %verify_binary_search		; <i32> [#uses=1]
	%tmp.156 = getelementptr %struct..s_annealing_sched, %struct..s_annealing_sched* %annealing_sched, i64 0, i32 0		; <i32*> [#uses=1]
	%tmp.157 = load i32, i32* %tmp.156		; <i32> [#uses=1]
	%tmp.158 = getelementptr %struct..s_annealing_sched, %struct..s_annealing_sched* %annealing_sched, i64 0, i32 1		; <float*> [#uses=1]
	%tmp.159 = load float, float* %tmp.158		; <float> [#uses=1]
	%tmp.160 = getelementptr %struct..s_annealing_sched, %struct..s_annealing_sched* %annealing_sched, i64 0, i32 2		; <float*> [#uses=1]
	%tmp.161 = load float, float* %tmp.160		; <float> [#uses=1]
	%tmp.162 = getelementptr %struct..s_annealing_sched, %struct..s_annealing_sched* %annealing_sched, i64 0, i32 3		; <float*> [#uses=1]
	%tmp.163 = load float, float* %tmp.162		; <float> [#uses=1]
	%tmp.164 = getelementptr %struct..s_annealing_sched, %struct..s_annealing_sched* %annealing_sched, i64 0, i32 4		; <float*> [#uses=1]
	%tmp.165 = load float, float* %tmp.164		; <float> [#uses=1]
	%tmp.166 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 0		; <float*> [#uses=1]
	%tmp.167 = load float, float* %tmp.166		; <float> [#uses=1]
	%tmp.168 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 1		; <float*> [#uses=1]
	%tmp.169 = load float, float* %tmp.168		; <float> [#uses=1]
	%tmp.170 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 2		; <float*> [#uses=1]
	%tmp.171 = load float, float* %tmp.170		; <float> [#uses=1]
	%tmp.172 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 3		; <float*> [#uses=1]
	%tmp.173 = load float, float* %tmp.172		; <float> [#uses=1]
	%tmp.174 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 4		; <float*> [#uses=1]
	%tmp.175 = load float, float* %tmp.174		; <float> [#uses=1]
	%tmp.176 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 5		; <i32*> [#uses=1]
	%tmp.177 = load i32, i32* %tmp.176		; <i32> [#uses=1]
	%tmp.178 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 6		; <i32*> [#uses=1]
	%tmp.179 = load i32, i32* %tmp.178		; <i32> [#uses=1]
	%tmp.181 = load i32, i32* %tmp.118		; <i32> [#uses=1]
	%tmp.182 = getelementptr %struct..s_router_opts, %struct..s_router_opts* %router_opts, i64 0, i32 8		; <i32*> [#uses=1]
	%tmp.183 = load i32, i32* %tmp.182		; <i32> [#uses=1]
	%tmp.184 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 0		; <i32*> [#uses=1]
	%tmp.185 = load i32, i32* %tmp.184		; <i32> [#uses=1]
	%tmp.186 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 1		; <float*> [#uses=1]
	%tmp.187 = load float, float* %tmp.186		; <float> [#uses=1]
	%tmp.188 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 2		; <float*> [#uses=1]
	%tmp.189 = load float, float* %tmp.188		; <float> [#uses=1]
	%tmp.190 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 3		; <float*> [#uses=1]
	%tmp.191 = load float, float* %tmp.190		; <float> [#uses=1]
	%tmp.192 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 4		; <i32*> [#uses=1]
	%tmp.193 = load i32, i32* %tmp.192		; <i32> [#uses=1]
	%tmp.194 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 5		; <i32*> [#uses=1]
	%tmp.195 = load i32, i32* %tmp.194		; <i32> [#uses=1]
	%tmp.196 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 6		; <i16*> [#uses=1]
	%tmp.197 = load i16, i16* %tmp.196		; <i16> [#uses=1]
	%tmp.198 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 7		; <i16*> [#uses=1]
	%tmp.199 = load i16, i16* %tmp.198		; <i16> [#uses=1]
	%tmp.200 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 8		; <i16*> [#uses=1]
	%tmp.201 = load i16, i16* %tmp.200		; <i16> [#uses=1]
	%tmp.202 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 9		; <float*> [#uses=1]
	%tmp.203 = load float, float* %tmp.202		; <float> [#uses=1]
	%tmp.204 = getelementptr %struct..s_det_routing_arch, %struct..s_det_routing_arch* %det_routing_arch, i64 0, i32 10		; <float*> [#uses=1]
	%tmp.205 = load float, float* %tmp.204		; <float> [#uses=1]
	%tmp.206 = load %struct..s_segment_inf*, %struct..s_segment_inf** %segment_inf		; <%struct..s_segment_inf*> [#uses=1]
	%tmp.208 = load i32, i32* %tmp.109		; <i32> [#uses=1]
	%tmp.209 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 1		; <float*> [#uses=1]
	%tmp.210 = load float, float* %tmp.209		; <float> [#uses=1]
	%tmp.211 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 2		; <float*> [#uses=1]
	%tmp.212 = load float, float* %tmp.211		; <float> [#uses=1]
	%tmp.213 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 3		; <float*> [#uses=1]
	%tmp.214 = load float, float* %tmp.213		; <float> [#uses=1]
	%tmp.215 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 4		; <float*> [#uses=1]
	%tmp.216 = load float, float* %tmp.215		; <float> [#uses=1]
	%tmp.217 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 5		; <float*> [#uses=1]
	%tmp.218 = load float, float* %tmp.217		; <float> [#uses=1]
	%tmp.219 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 6		; <float*> [#uses=1]
	%tmp.220 = load float, float* %tmp.219		; <float> [#uses=1]
	%tmp.221 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 7		; <float*> [#uses=1]
	%tmp.222 = load float, float* %tmp.221		; <float> [#uses=1]
	%tmp.223 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 8		; <float*> [#uses=1]
	%tmp.224 = load float, float* %tmp.223		; <float> [#uses=1]
	%tmp.225 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 9		; <float*> [#uses=1]
	%tmp.226 = load float, float* %tmp.225		; <float> [#uses=1]
	%tmp.227 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, { i32, float, float, float, float, float, float, float, float, float, float }* %timing_inf, i64 0, i32 10		; <float*> [#uses=1]
	%tmp.228 = load float, float* %tmp.227		; <float> [#uses=1]
	call void @place_and_route( i32 %tmp.135, i32 %tmp.137, float %tmp.139, i32 %tmp.141, i32 %tmp.143, i8* %tmp.145, i32 %tmp.147, i32 %tmp.149, i8* %tmp.107, i8* %tmp.105, i8* %tmp.106, i8* %tmp.108, i32 %tmp.154, i32 %tmp.155, i32 %tmp.157, float %tmp.159, float %tmp.161, float %tmp.163, float %tmp.165, float %tmp.167, float %tmp.169, float %tmp.171, float %tmp.173, float %tmp.175, i32 %tmp.177, i32 %tmp.179, i32 %tmp.181, i32 %tmp.183, i32 %tmp.185, float %tmp.187, float %tmp.189, float %tmp.191, i32 %tmp.193, i32 %tmp.195, i16 %tmp.197, i16 %tmp.199, i16 %tmp.201, float %tmp.203, float %tmp.205, %struct..s_segment_inf* %tmp.206, i32 %tmp.208, float %tmp.210, float %tmp.212, float %tmp.214, float %tmp.216, float %tmp.218, float %tmp.220, float %tmp.222, float %tmp.224, float %tmp.226, float %tmp.228 )
	%tmp.231 = load i32, i32* %show_graphics		; <i32> [#uses=1]
	%tmp.232 = icmp ne i32 %tmp.231, 0		; <i1> [#uses=1]
	br i1 %tmp.232, label %then.2, label %endif.2

then.2:		; preds = %entry
	br label %endif.2

endif.2:		; preds = %then.2, %entry
	ret i32 0
}

declare i32 @printf(i8*, ...)

declare void @place_and_route(i32, i32, float, i32, i32, i8*, i32, i32, i8*, i8*, i8*, i8*, i32, i32, i32, float, float, float, float, float, float, float, float, float, i32, i32, i32, i32, i32, float, float, float, i32, i32, i16, i16, i16, float, float, %struct..s_segment_inf*, i32, float, float, float, float, float, float, float, float, float, float)

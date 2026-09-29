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
	%struct..s_placer_opts = type { i32, float, i32, i32, ptr, i32, i32 }
	%struct..s_router_opts = type { float, float, float, float, float, i32, i32, i32, i32 }
	%struct..s_segment_inf = type { float, i32, i16, i16, float, float, i32, float, float }
	%struct..s_switch_inf = type { i32, float, float, float, float }

define i32 @main(i32 %argc.1, ptr %argv.1) {
entry:
	%net_file = alloca [300 x i8]		; <ptr> [#uses=1]
	%place_file = alloca [300 x i8]		; <ptr> [#uses=1]
	%arch_file = alloca [300 x i8]		; <ptr> [#uses=1]
	%route_file = alloca [300 x i8]		; <ptr> [#uses=1]
	%full_stats = alloca i32		; <ptr> [#uses=1]
	%operation = alloca i32		; <ptr> [#uses=1]
	%verify_binary_search = alloca i32		; <ptr> [#uses=1]
	%show_graphics = alloca i32		; <ptr> [#uses=1]
	%annealing_sched = alloca %struct..s_annealing_sched		; <ptr> [#uses=5]
	%placer_opts = alloca %struct..s_placer_opts		; <ptr> [#uses=7]
	%router_opts = alloca %struct..s_router_opts		; <ptr> [#uses=9]
	%det_routing_arch = alloca %struct..s_det_routing_arch		; <ptr> [#uses=11]
	%segment_inf = alloca ptr		; <ptr> [#uses=1]
	%timing_inf = alloca { i32, float, float, float, float, float, float, float, float, float, float }		; <ptr> [#uses=11]
	%tmp.101 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 4		; <ptr> [#uses=1]
	%tmp.105 = getelementptr [300 x i8], ptr %net_file, i64 0, i64 0		; <ptr> [#uses=1]
	%tmp.106 = getelementptr [300 x i8], ptr %arch_file, i64 0, i64 0		; <ptr> [#uses=1]
	%tmp.107 = getelementptr [300 x i8], ptr %place_file, i64 0, i64 0		; <ptr> [#uses=1]
	%tmp.108 = getelementptr [300 x i8], ptr %route_file, i64 0, i64 0		; <ptr> [#uses=1]
	%tmp.109 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.112 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.114 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 6		; <ptr> [#uses=1]
	%tmp.118 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 7		; <ptr> [#uses=1]
	%tmp.135 = load i32, ptr %operation		; <i32> [#uses=1]
	%tmp.137 = load i32, ptr %tmp.112		; <i32> [#uses=1]
	%tmp.138 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.139 = load float, ptr %tmp.138		; <float> [#uses=1]
	%tmp.140 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 2		; <ptr> [#uses=1]
	%tmp.141 = load i32, ptr %tmp.140		; <i32> [#uses=1]
	%tmp.142 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 3		; <ptr> [#uses=1]
	%tmp.143 = load i32, ptr %tmp.142		; <i32> [#uses=1]
	%tmp.145 = load ptr, ptr %tmp.101		; <ptr> [#uses=1]
	%tmp.146 = getelementptr %struct..s_placer_opts, ptr %placer_opts, i64 0, i32 5		; <ptr> [#uses=1]
	%tmp.147 = load i32, ptr %tmp.146		; <i32> [#uses=1]
	%tmp.149 = load i32, ptr %tmp.114		; <i32> [#uses=1]
	%tmp.154 = load i32, ptr %full_stats		; <i32> [#uses=1]
	%tmp.155 = load i32, ptr %verify_binary_search		; <i32> [#uses=1]
	%tmp.156 = getelementptr %struct..s_annealing_sched, ptr %annealing_sched, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.157 = load i32, ptr %tmp.156		; <i32> [#uses=1]
	%tmp.158 = getelementptr %struct..s_annealing_sched, ptr %annealing_sched, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.159 = load float, ptr %tmp.158		; <float> [#uses=1]
	%tmp.160 = getelementptr %struct..s_annealing_sched, ptr %annealing_sched, i64 0, i32 2		; <ptr> [#uses=1]
	%tmp.161 = load float, ptr %tmp.160		; <float> [#uses=1]
	%tmp.162 = getelementptr %struct..s_annealing_sched, ptr %annealing_sched, i64 0, i32 3		; <ptr> [#uses=1]
	%tmp.163 = load float, ptr %tmp.162		; <float> [#uses=1]
	%tmp.164 = getelementptr %struct..s_annealing_sched, ptr %annealing_sched, i64 0, i32 4		; <ptr> [#uses=1]
	%tmp.165 = load float, ptr %tmp.164		; <float> [#uses=1]
	%tmp.166 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.167 = load float, ptr %tmp.166		; <float> [#uses=1]
	%tmp.168 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.169 = load float, ptr %tmp.168		; <float> [#uses=1]
	%tmp.170 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 2		; <ptr> [#uses=1]
	%tmp.171 = load float, ptr %tmp.170		; <float> [#uses=1]
	%tmp.172 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 3		; <ptr> [#uses=1]
	%tmp.173 = load float, ptr %tmp.172		; <float> [#uses=1]
	%tmp.174 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 4		; <ptr> [#uses=1]
	%tmp.175 = load float, ptr %tmp.174		; <float> [#uses=1]
	%tmp.176 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 5		; <ptr> [#uses=1]
	%tmp.177 = load i32, ptr %tmp.176		; <i32> [#uses=1]
	%tmp.178 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 6		; <ptr> [#uses=1]
	%tmp.179 = load i32, ptr %tmp.178		; <i32> [#uses=1]
	%tmp.181 = load i32, ptr %tmp.118		; <i32> [#uses=1]
	%tmp.182 = getelementptr %struct..s_router_opts, ptr %router_opts, i64 0, i32 8		; <ptr> [#uses=1]
	%tmp.183 = load i32, ptr %tmp.182		; <i32> [#uses=1]
	%tmp.184 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.185 = load i32, ptr %tmp.184		; <i32> [#uses=1]
	%tmp.186 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.187 = load float, ptr %tmp.186		; <float> [#uses=1]
	%tmp.188 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 2		; <ptr> [#uses=1]
	%tmp.189 = load float, ptr %tmp.188		; <float> [#uses=1]
	%tmp.190 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 3		; <ptr> [#uses=1]
	%tmp.191 = load float, ptr %tmp.190		; <float> [#uses=1]
	%tmp.192 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 4		; <ptr> [#uses=1]
	%tmp.193 = load i32, ptr %tmp.192		; <i32> [#uses=1]
	%tmp.194 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 5		; <ptr> [#uses=1]
	%tmp.195 = load i32, ptr %tmp.194		; <i32> [#uses=1]
	%tmp.196 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 6		; <ptr> [#uses=1]
	%tmp.197 = load i16, ptr %tmp.196		; <i16> [#uses=1]
	%tmp.198 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 7		; <ptr> [#uses=1]
	%tmp.199 = load i16, ptr %tmp.198		; <i16> [#uses=1]
	%tmp.200 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 8		; <ptr> [#uses=1]
	%tmp.201 = load i16, ptr %tmp.200		; <i16> [#uses=1]
	%tmp.202 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 9		; <ptr> [#uses=1]
	%tmp.203 = load float, ptr %tmp.202		; <float> [#uses=1]
	%tmp.204 = getelementptr %struct..s_det_routing_arch, ptr %det_routing_arch, i64 0, i32 10		; <ptr> [#uses=1]
	%tmp.205 = load float, ptr %tmp.204		; <float> [#uses=1]
	%tmp.206 = load ptr, ptr %segment_inf		; <ptr> [#uses=1]
	%tmp.208 = load i32, ptr %tmp.109		; <i32> [#uses=1]
	%tmp.209 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 1		; <ptr> [#uses=1]
	%tmp.210 = load float, ptr %tmp.209		; <float> [#uses=1]
	%tmp.211 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 2		; <ptr> [#uses=1]
	%tmp.212 = load float, ptr %tmp.211		; <float> [#uses=1]
	%tmp.213 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 3		; <ptr> [#uses=1]
	%tmp.214 = load float, ptr %tmp.213		; <float> [#uses=1]
	%tmp.215 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 4		; <ptr> [#uses=1]
	%tmp.216 = load float, ptr %tmp.215		; <float> [#uses=1]
	%tmp.217 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 5		; <ptr> [#uses=1]
	%tmp.218 = load float, ptr %tmp.217		; <float> [#uses=1]
	%tmp.219 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 6		; <ptr> [#uses=1]
	%tmp.220 = load float, ptr %tmp.219		; <float> [#uses=1]
	%tmp.221 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 7		; <ptr> [#uses=1]
	%tmp.222 = load float, ptr %tmp.221		; <float> [#uses=1]
	%tmp.223 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 8		; <ptr> [#uses=1]
	%tmp.224 = load float, ptr %tmp.223		; <float> [#uses=1]
	%tmp.225 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 9		; <ptr> [#uses=1]
	%tmp.226 = load float, ptr %tmp.225		; <float> [#uses=1]
	%tmp.227 = getelementptr { i32, float, float, float, float, float, float, float, float, float, float }, ptr %timing_inf, i64 0, i32 10		; <ptr> [#uses=1]
	%tmp.228 = load float, ptr %tmp.227		; <float> [#uses=1]
	call void @place_and_route( i32 %tmp.135, i32 %tmp.137, float %tmp.139, i32 %tmp.141, i32 %tmp.143, ptr %tmp.145, i32 %tmp.147, i32 %tmp.149, ptr %tmp.107, ptr %tmp.105, ptr %tmp.106, ptr %tmp.108, i32 %tmp.154, i32 %tmp.155, i32 %tmp.157, float %tmp.159, float %tmp.161, float %tmp.163, float %tmp.165, float %tmp.167, float %tmp.169, float %tmp.171, float %tmp.173, float %tmp.175, i32 %tmp.177, i32 %tmp.179, i32 %tmp.181, i32 %tmp.183, i32 %tmp.185, float %tmp.187, float %tmp.189, float %tmp.191, i32 %tmp.193, i32 %tmp.195, i16 %tmp.197, i16 %tmp.199, i16 %tmp.201, float %tmp.203, float %tmp.205, ptr %tmp.206, i32 %tmp.208, float %tmp.210, float %tmp.212, float %tmp.214, float %tmp.216, float %tmp.218, float %tmp.220, float %tmp.222, float %tmp.224, float %tmp.226, float %tmp.228 )
	%tmp.231 = load i32, ptr %show_graphics		; <i32> [#uses=1]
	%tmp.232 = icmp ne i32 %tmp.231, 0		; <i1> [#uses=1]
	br i1 %tmp.232, label %then.2, label %endif.2

then.2:		; preds = %entry
	br label %endif.2

endif.2:		; preds = %then.2, %entry
	ret i32 0
}

declare i32 @printf(ptr, ...)

declare void @place_and_route(i32, i32, float, i32, i32, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32, i32, float, float, float, float, float, float, float, float, float, i32, i32, i32, i32, i32, float, float, float, i32, i32, i16, i16, i16, float, float, ptr, i32, float, float, float, float, float, float, float, float, float, float)

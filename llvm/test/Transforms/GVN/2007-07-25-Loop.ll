; RUN: opt < %s -passes=gvn | llvm-dis

	%struct.s_segment_inf = type { float, i32, i16, i16, float, float, i32, float, float }

define void @print_arch(i8* %arch_file, i32 %route_type, i64 %det_routing_arch.0.0, i64 %det_routing_arch.0.1, i64 %det_routing_arch.0.2, i64 %det_routing_arch.0.3, i64 %det_routing_arch.0.4, %struct.s_segment_inf* %segment_inf, i64 %timing_inf.0.0, i64 %timing_inf.0.1, i64 %timing_inf.0.2, i64 %timing_inf.0.3, i64 %timing_inf.0.4, i32 %timing_inf.1) {
entry:
	br i1 false, label %bb278, label %bb344

bb278:		; preds = %bb278, %entry
	br i1 false, label %bb278, label %bb344

bb344:		; preds = %bb278, %entry
	%tmp38758 = load i16, i16* null, align 2		; <i16> [#uses=0]
	ret void
}

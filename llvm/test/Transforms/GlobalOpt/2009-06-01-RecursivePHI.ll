; RUN: opt < %s -passes=globalopt
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

	%struct.s_annealing_sched = type { i32, float, float, float, float }
	%struct.s_bb = type { i32, i32, i32, i32 }
	%struct.s_net = type { ptr, i32, ptr, float, float }
	%struct.s_placer_opts = type { i32, float, i32, i32, ptr, i32, i32 }
@net = internal global ptr null		; <ptr> [#uses=4]

define fastcc void @alloc_and_load_placement_structs(i32 %place_cost_type, i32 %num_regions, float %place_cost_exp, ptr nocapture %old_region_occ_x, ptr nocapture %old_region_occ_y, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5, i1 %c6, i1 %c7, i1 %c8, i1 %c9, i1 %c10, i1 %c11, i1 %c12) nounwind ssp {
entry:
	br i1 %c1, label %bb.i, label %my_malloc.exit

bb.i:		; preds = %entry
	unreachable

my_malloc.exit:		; preds = %entry
	br i1 %c2, label %bb.i81, label %my_malloc.exit83

bb.i81:		; preds = %my_malloc.exit
	unreachable

my_malloc.exit83:		; preds = %my_malloc.exit
	br i1 %c3, label %bb.i.i57, label %my_calloc.exit.i

bb.i.i57:		; preds = %my_malloc.exit83
	unreachable

my_calloc.exit.i:		; preds = %my_malloc.exit83
	br i1 %c4, label %bb.i4.i, label %my_calloc.exit5.i

bb.i4.i:		; preds = %my_calloc.exit.i
	unreachable

my_calloc.exit5.i:		; preds = %my_calloc.exit.i
	%.pre.i58 = load ptr, ptr @net, align 4		; <ptr> [#uses=1]
	br label %bb17.i78

bb1.i61:		; preds = %bb4.preheader.i, %bb1.i61
	br i1 %c5, label %bb1.i61, label %bb5.i62

bb5.i62:		; preds = %bb1.i61
	br i1 %c6, label %bb6.i64, label %bb15.preheader.i

bb15.preheader.i:		; preds = %bb4.preheader.i, %bb5.i62
	br label %bb16.i77

bb6.i64:		; preds = %bb5.i62
	br i1 %c7, label %bb7.i65, label %bb8.i67

bb7.i65:		; preds = %bb6.i64
	unreachable

bb8.i67:		; preds = %bb6.i64
	br i1 %c8, label %bb.i1.i68, label %my_malloc.exit.i70

bb.i1.i68:		; preds = %bb8.i67
	unreachable

my_malloc.exit.i70:		; preds = %bb8.i67
	%0 = load ptr, ptr @net, align 4		; <ptr> [#uses=1]
	br i1 %c9, label %bb9.i71, label %bb16.i77

bb9.i71:		; preds = %bb9.i71, %my_malloc.exit.i70
	%1 = load ptr, ptr @net, align 4		; <ptr> [#uses=1]
	br i1 %c10, label %bb9.i71, label %bb16.i77

bb16.i77:		; preds = %bb9.i71, %my_malloc.exit.i70, %bb15.preheader.i
	%.pre41.i.rle244 = phi ptr [ %.pre41.i, %bb15.preheader.i ], [ %0, %my_malloc.exit.i70 ], [ %1, %bb9.i71 ]		; <ptr> [#uses=1]
	br label %bb17.i78

bb17.i78:		; preds = %bb16.i77, %my_calloc.exit5.i
	%.pre41.i = phi ptr [ %.pre41.i.rle244, %bb16.i77 ], [ %.pre.i58, %my_calloc.exit5.i ]		; <ptr> [#uses=1]
	br i1 %c11, label %bb4.preheader.i, label %alloc_and_load_unique_pin_list.exit

bb4.preheader.i:		; preds = %bb17.i78
	br i1 %c12, label %bb1.i61, label %bb15.preheader.i

alloc_and_load_unique_pin_list.exit:		; preds = %bb17.i78
	ret void
}

define void @read_net(ptr %net_file, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5) nounwind ssp {
entry:
	br i1 %c1, label %bb3.us.us.i, label %bb6.preheader

bb6.preheader:		; preds = %entry
	br i1 %c2, label %bb7, label %bb

bb3.us.us.i:		; preds = %entry
	unreachable

bb:		; preds = %bb6.preheader
	br i1 %c3, label %bb.i34, label %bb1.i38

bb.i34:		; preds = %bb
	unreachable

bb1.i38:		; preds = %bb
	%mallocsize = mul i64 28, undef                  ; <i64> [#uses=1]
	%malloccall = tail call ptr @malloc(i64 %mallocsize)      ; <ptr> [#uses=1]
	br i1 %c4, label %bb.i1.i39, label %my_malloc.exit2.i

bb.i1.i39:		; preds = %bb1.i38
	unreachable

my_malloc.exit2.i:		; preds = %bb1.i38
	store ptr %malloccall, ptr @net, align 4
	br i1 %c5, label %bb.i7.i40, label %my_malloc.exit8.i

bb.i7.i40:		; preds = %my_malloc.exit2.i
	unreachable

my_malloc.exit8.i:		; preds = %my_malloc.exit2.i
	unreachable

bb7:		; preds = %bb6.preheader
	unreachable
}

declare noalias ptr @malloc(i64)

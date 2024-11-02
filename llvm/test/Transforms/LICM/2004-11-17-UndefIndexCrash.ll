; RUN: opt < %s -licm -disable-output
	%struct.roadlet = type { ptr, ptr, [8 x ptr], [8 x ptr] }
	%struct.vehicle = type { ptr, ptr, i32, i32, %union.._631., i32 }
	%union.._631. = type { i32 }

declare ptr @_Z11return_nullP7roadletP7vehicle9direction(ptr, ptr, i32)

declare ptr @_Z14lane_switch_okP7roadletP7vehicle9direction(ptr, ptr, i32)

define void @main() {
__main.entry:
	br label %invoke_cont.3
invoke_cont.3:		; preds = %invoke_cont.3, %__main.entry
	%tmp.34.i.i502.7 = getelementptr %struct.roadlet, ptr null, i32 0, i32 3, i32 7		; <ptr> [#uses=1]
	store ptr @_Z11return_nullP7roadletP7vehicle9direction, ptr %tmp.34.i.i502.7
	store ptr @_Z14lane_switch_okP7roadletP7vehicle9direction, ptr null
	%tmp.4.i.i339 = getelementptr %struct.roadlet, ptr null, i32 0, i32 3, i32 undef		; <ptr> [#uses=1]
	store ptr @_Z11return_nullP7roadletP7vehicle9direction, ptr %tmp.4.i.i339
	br label %invoke_cont.3
}

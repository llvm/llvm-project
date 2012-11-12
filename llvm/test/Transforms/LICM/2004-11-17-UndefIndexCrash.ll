; RUN: opt < %s -licm -disable-output
	%struct.roadlet = type { i8*, %struct.vehicle*, [8 x %struct.roadlet*], [8 x %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)*] }
	%struct.vehicle = type { %struct.roadlet*, i8*, i32, i32, %union.._631., i32 }
	%union.._631. = type { i32 }

declare %struct.roadlet* @_Z11return_nullP7roadletP7vehicle9direction(%struct.roadlet*, %struct.vehicle*, i32)

declare %struct.roadlet* @_Z14lane_switch_okP7roadletP7vehicle9direction(%struct.roadlet*, %struct.vehicle*, i32)

define void @main() {
__main.entry:
	br label %invoke_cont.3
invoke_cont.3:		; preds = %invoke_cont.3, %__main.entry
	%tmp.34.i.i502.7 = getelementptr %struct.roadlet* null, i32 0, i32 3, i32 7		; <%struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)**> [#uses=1]
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)* @_Z11return_nullP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)** %tmp.34.i.i502.7
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)* @_Z14lane_switch_okP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)** null
	%tmp.4.i.i339 = getelementptr %struct.roadlet* null, i32 0, i32 3, i32 undef		; <%struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)**> [#uses=1]
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)* @_Z11return_nullP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, i32)** %tmp.4.i.i339
	br label %invoke_cont.3
}

; RUN: llc < %s -march=arm

	%struct.hit_t = type { %struct.v_t, double }
	%struct.node_t = type { %struct.hit_t, %struct.hit_t, i32 }
	%struct.v_t = type { double, double, double }

define fastcc %struct.node_t* @_ZL6createP6node_tii3v_tS1_d(%struct.node_t* %n, i32 %lvl, i32 %dist, i64 %c.0.0, i64 %c.0.1, i64 %c.0.2, i64 %d.0.0, i64 %d.0.1, i64 %d.0.2, double %r) nounwind {
entry:
	%0 = getelementptr %struct.node_t* %n, i32 0, i32 1		; <%struct.hit_t*> [#uses=1]
	%1 = bitcast %struct.hit_t* %0 to i256*		; <i256*> [#uses=1]
	store i256 0, i256* %1, align 4
	unreachable
}

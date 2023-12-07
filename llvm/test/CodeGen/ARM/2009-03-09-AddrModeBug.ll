; RUN: llc -mtriple=arm-eabi %s -o /dev/null

	%struct.hit_t = type { %struct.v_t, double }
	%struct.node_t = type { %struct.hit_t, %struct.hit_t, i32 }
	%struct.v_t = type { double, double, double }

define fastcc ptr @_ZL6createP6node_tii3v_tS1_d(ptr %n, i32 %lvl, i32 %dist, i64 %c.0.0, i64 %c.0.1, i64 %c.0.2, i64 %d.0.0, i64 %d.0.1, i64 %d.0.2, double %r) nounwind {
entry:
	%0 = getelementptr %struct.node_t, ptr %n, i32 0, i32 1		; <ptr> [#uses=1]
	store i256 0, ptr %0, align 4
	unreachable
}

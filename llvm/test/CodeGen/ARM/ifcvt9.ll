; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define fastcc void @t() nounwind {
entry:
	br i1 undef, label %bb.i.i3, label %growMapping.exit

bb.i.i3:		; preds = %entry
	unreachable

growMapping.exit:		; preds = %entry
	unreachable
}

; RUN: llvm-as < %s | llc -march=x86 | not grep mov

	%struct.active_line = type { %struct.gs_fixed_point, %struct.gs_fixed_point, i32, i32, i32, %struct.line_segment*, i32, i16, i16, %struct.active_line*, %struct.active_line* }
	%struct.gs_fixed_point = type { i32, i32 }
	%struct.line_list = type { %struct.active_line*, i32, i16, %struct.active_line*, %struct.active_line*, %struct.active_line*, %struct.active_line, i32 }
	%struct.line_segment = type { %struct.line_segment*, %struct.line_segment*, i32, %struct.gs_fixed_point }
	%struct.subpath = type { %struct.line_segment*, %struct.line_segment*, i32, %struct.gs_fixed_point, %struct.line_segment*, i32, i32, i8 }

define fastcc void @add_y_list(%struct.subpath* %ppath.0.4.val, i16 signext  %tag, %struct.line_list* %ll, i32 %pbox.0.0.1.val, i32 %pbox.0.1.0.val, i32 %pbox.0.1.1.val) nounwind  {
entry:
	br i1 false, label %return, label %bb
bb:		; preds = %bb280, %entry
	%psub.1.reg2mem.0 = phi %struct.subpath* [ %psub.0.reg2mem.0, %bb280 ], [ undef, %entry ]		; <%struct.subpath*> [#uses=1]
	%plast.1.reg2mem.0 = phi %struct.line_segment* [ %plast.0.reg2mem.0, %bb280 ], [ undef, %entry ]		; <%struct.line_segment*> [#uses=1]
	%prev_dir.0.reg2mem.0 = phi i32 [ %dir.0.reg2mem.0, %bb280 ], [ undef, %entry ]		; <i32> [#uses=1]
	br i1 false, label %bb280, label %bb109
bb109:		; preds = %bb
	%tmp113 = icmp sgt i32 0, %prev_dir.0.reg2mem.0		; <i1> [#uses=1]
	br i1 %tmp113, label %bb116, label %bb280
bb116:		; preds = %bb109
	ret void
bb280:		; preds = %bb109, %bb
	%psub.0.reg2mem.0 = phi %struct.subpath* [ null, %bb ], [ %psub.1.reg2mem.0, %bb109 ]		; <%struct.subpath*> [#uses=1]
	%plast.0.reg2mem.0 = phi %struct.line_segment* [ null, %bb ], [ %plast.1.reg2mem.0, %bb109 ]		; <%struct.line_segment*> [#uses=1]
	%dir.0.reg2mem.0 = phi i32 [ 0, %bb ], [ 0, %bb109 ]		; <i32> [#uses=1]
	br i1 false, label %return, label %bb
return:		; preds = %bb280, %entry
	ret void
}

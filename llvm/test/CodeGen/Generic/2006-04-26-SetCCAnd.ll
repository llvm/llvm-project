; RUN: llvm-as < %s | llc
; PR748
@G = external global i16		; <i16*> [#uses=1]

define void @OmNewObjHdr() {
entry:
	br i1 false, label %endif.4, label %then.0

then.0:		; preds = %entry
	ret void

endif.4:		; preds = %entry
	br i1 false, label %else.3, label %shortcirc_next.3

shortcirc_next.3:		; preds = %endif.4
	ret void

else.3:		; preds = %endif.4
	switch i32 0, label %endif.10 [
		 i32 5001, label %then.10
		 i32 -5008, label %then.10
	]

then.10:		; preds = %else.3, %else.3
	%tmp.112 = load i16* null		; <i16> [#uses=2]
	%tmp.113 = load i16* @G		; <i16> [#uses=2]
	%tmp.114 = icmp ugt i16 %tmp.112, %tmp.113		; <i1> [#uses=1]
	%tmp.120 = icmp ult i16 %tmp.112, %tmp.113		; <i1> [#uses=1]
	%bothcond = and i1 %tmp.114, %tmp.120		; <i1> [#uses=1]
	br i1 %bothcond, label %else.4, label %then.11

then.11:		; preds = %then.10
	ret void

else.4:		; preds = %then.10
	ret void

endif.10:		; preds = %else.3
	ret void
}

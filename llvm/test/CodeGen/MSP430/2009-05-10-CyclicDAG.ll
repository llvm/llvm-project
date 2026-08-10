; RUN: llc < %s
; PR4136

target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-unknown-linux-gnu"
@uip_len = external global i16		; <i16*> [#uses=2]

define void @uip_arp_arpin() nounwind {
entry:
	%tmp = load volatile i16, ptr @uip_len		; <i16> [#uses=1]
	%cmp = icmp ult i16 %tmp, 42		; <i1> [#uses=1]
	store volatile i16 0, ptr @uip_len
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	ret void

if.end:		; preds = %entry
	switch i16 0, label %return [
		i16 256, label %sw.bb
		i16 512, label %sw.bb18
	]

sw.bb:		; preds = %if.end
	ret void

sw.bb18:		; preds = %if.end
	ret void

return:		; preds = %if.end
	ret void
}

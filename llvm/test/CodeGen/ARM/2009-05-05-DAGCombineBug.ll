; RUN: llc < %s -mtriple=arm-unknown-linux-gnueabi -mattr=+v6
; PR4166

	%"byte[]" = type { i32, ptr }
	%tango.time.Time.Time = type { i64 }

define fastcc void @t() {
entry:
	%tmp28 = call fastcc i1 null(ptr null, %"byte[]" undef, %"byte[]" undef, ptr byval(%tango.time.Time.Time) null)		; <i1> [#uses=0]
	ret void
}

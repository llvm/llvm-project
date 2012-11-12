; RUN: llc < %s -mtriple=arm-unknown-linux-gnueabi -mattr=+v6
; PR4166

	%"byte[]" = type { i32, i8* }
	%tango.time.Time.Time = type { i64 }

define fastcc void @t() {
entry:
	%tmp28 = call fastcc i1 null(i32* null, %"byte[]" undef, %"byte[]" undef, %tango.time.Time.Time* byval null)		; <i1> [#uses=0]
	ret void
}

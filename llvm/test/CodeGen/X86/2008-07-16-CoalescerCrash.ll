; RUN: llc < %s -mtriple=x86_64-apple-darwin

	%struct.SV = type { ptr, i64, i64 }
@"\01LC25" = external constant [8 x i8]		; <ptr> [#uses=1]

declare void @Perl_sv_catpvf(ptr, ptr, ...) nounwind 

declare fastcc i64 @Perl_utf8n_to_uvuni(ptr, i64, ptr, i64) nounwind 

define fastcc ptr @Perl_pv_uni_display(ptr %dsv, ptr %spv, i64 %len, i64 %pvlim, i64 %flags) nounwind  {
entry:
	br i1 false, label %bb, label %bb40

bb:		; preds = %entry
	tail call fastcc i64 @Perl_utf8n_to_uvuni( ptr null, i64 13, ptr null, i64 255 ) nounwind 		; <i64>:0 [#uses=1]
	br i1 false, label %bb6, label %bb33

bb6:		; preds = %bb
	br i1 false, label %bb30, label %bb31

bb30:		; preds = %bb6
	unreachable

bb31:		; preds = %bb6
	icmp eq i8 0, 0		; <i1>:1 [#uses=0]
	br label %bb33

bb33:		; preds = %bb31, %bb
	tail call void (ptr, ptr, ...) @Perl_sv_catpvf( ptr %dsv, ptr @"\01LC25", i64 %0 ) nounwind 
	unreachable

bb40:		; preds = %entry
	ret ptr null
}

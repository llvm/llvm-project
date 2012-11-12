; RUN: opt < %s -inline -simplify-libcalls -functionattrs | \
; RUN:   llvm-dis | grep nocapture | count 2
; Check that nocapture attributes are added when run after an SCC pass.
; PR3520

define i32 @use(i8* %x) nounwind readonly {
entry:
	%0 = tail call i64 @strlen(i8* %x) nounwind readonly		; <i64> [#uses=1]
	%1 = trunc i64 %0 to i32		; <i32> [#uses=1]
	ret i32 %1
}

declare i64 @strlen(i8*) nounwind readonly

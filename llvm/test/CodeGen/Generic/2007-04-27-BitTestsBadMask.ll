; RUN: llvm-as < %s | llc -march=x86 | grep je | count 3
; RUN: llvm-as < %s | llc -march=x86-64 | grep 4297064449
; PR 1325+

define i32 @foo(i8 %bar) {
entry:
	switch i8 %bar, label %bb1203 [
		 i8 117, label %bb1204
		 i8 85, label %bb1204
		 i8 106, label %bb1204
	]

bb1203:		; preds = %entry
	ret i32 1

bb1204:		; preds = %entry, %entry, %entry
	ret i32 2
}

; RUN: llc < %s -relocation-model=pic -mtriple=i386-linux-gnu -asm-verbose=false \
; RUN:   | FileCheck %s --check-prefix=CHECK-LINUX
; RUN: llc < %s -relocation-model=pic -mark-data-regions -mtriple=i686-apple-darwin -asm-verbose=false \
; RUN:   | FileCheck %s --check-prefix=CHECK-DATA
; RUN: llc < %s -relocation-model=pic -mtriple=i686-apple-darwin -asm-verbose=false \
; RUN:   | FileCheck %s --check-prefix=CHECK-DATA
; RUN: llc < %s                       -mtriple=x86_64-apple-darwin | not grep 'lJTI'
; rdar://6971437
; rdar://7738756

declare void @_Z3bari(i32)

; CHECK-LINUX: _Z3fooILi1EEvi:
define linkonce void @_Z3fooILi1EEvi(i32 %Y) nounwind {
entry:
; CHECK:       L0$pb
; CHECK-NOT:   leal
; CHECK:       Ltmp0 = LJTI0_0-L0$pb
; CHECK-NEXT:  addl Ltmp0(%eax,%ecx,4)
; CHECK-NEXT:  jmpl *%eax

;; When data-in-code markers are enabled, we should see them around the jump
;; table.
; CHECK-DATA: .data_region jt32
; CHECK-DATA: LJTI0_0
; CHECK-DATA: .end_data_region

;; When they're not enabled, make sure we don't see them at all.
; CHECK-NOT: .data_region
; CHECK-LINUX-NOT: .data_region
	%Y_addr = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %Y, i32* %Y_addr
	%tmp = load i32, i32* %Y_addr		; <i32> [#uses=1]
	switch i32 %tmp, label %bb10 [
		 i32 0, label %bb3
		 i32 1, label %bb
		 i32 2, label %bb
		 i32 3, label %bb
		 i32 4, label %bb
		 i32 5, label %bb
		 i32 6, label %bb
		 i32 7, label %bb
		 i32 8, label %bb
		 i32 9, label %bb
		 i32 10, label %bb
		 i32 12, label %bb1
		 i32 13, label %bb5
		 i32 14, label %bb6
		 i32 16, label %bb2
		 i32 17, label %bb4
		 i32 23, label %bb8
		 i32 27, label %bb7
		 i32 34, label %bb9
	]

bb:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	call void @_Z3bari( i32 0 )
	br label %bb1

bb1:		; preds = %bb, %entry
	call void @_Z3bari( i32 1 )
	br label %bb2

bb2:		; preds = %bb1, %entry
	call void @_Z3bari( i32 2 )
	br label %bb11

bb3:		; preds = %entry
	br label %bb4

bb4:		; preds = %bb3, %entry
	br label %bb5

bb5:		; preds = %bb4, %entry
	br label %bb6

bb6:		; preds = %bb5, %entry
	call void @_Z3bari( i32 2 )
	br label %bb11

bb7:		; preds = %entry
	br label %bb8

bb8:		; preds = %bb7, %entry
	br label %bb9

bb9:		; preds = %bb8, %entry
	call void @_Z3bari( i32 3 )
	br label %bb11

bb10:		; preds = %entry
	br label %bb11

bb11:		; preds = %bb10, %bb9, %bb6, %bb2
	br label %return

return:		; preds = %bb11
	ret void
}

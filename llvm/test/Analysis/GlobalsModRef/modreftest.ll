; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='require<globals-aa>,gvn' -S | FileCheck %s

@X = internal global i32 4		; <ptr> [#uses=2]

define i32 @test(ptr %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 12, ptr @X
; CHECK-NEXT: call void @doesnotmodX()
; CHECK-NEXT: ret i32 12
	store i32 12, ptr @X
	call void @doesnotmodX( )
	%V = load i32, ptr @X		; <i32> [#uses=1]
	ret i32 %V
}

define void @doesnotmodX() {
	ret void
}

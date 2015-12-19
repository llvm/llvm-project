; RUN: opt < %s -basicaa -globals-aa -gvn -S | FileCheck %s

@X = internal global i32 4		; <i32*> [#uses=2]

define i32 @test(i32* %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: call void @doesnotmodX()
; CHECK-NEXT: ret i32 12
	store i32 12, i32* @X
	call void @doesnotmodX( )
	%V = load i32, i32* @X		; <i32> [#uses=1]
	ret i32 %V
}

define void @doesnotmodX() {
	ret void
}

declare void @InaccessibleMemOnlyFunc( ) #0
declare void @InaccessibleMemOrArgMemOnlyFunc( ) #1

define i32 @test2(i32* %P) {
; CHECK:      @test2
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: call void @InaccessibleMemOnlyFunc()
; CHECK-NEXT: call void @InaccessibleMemOrArgMemOnlyFunc()
; CHECK-NOT:  load i32
; CHECK-NEXT: ret i32 12
	store i32 12, i32* @X
	call void @InaccessibleMemOnlyFunc( )
        call void @InaccessibleMemOrArgMemOnlyFunc( )
	%V = load i32, i32* @X		; <i32> [#uses=1]
	ret i32 %V
}

attributes #0 = { inaccessiblememonly }
attributes #1 = { inaccessiblemem_or_argmemonly }

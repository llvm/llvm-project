; RUN: opt < %s -basicaa -globals-aa -gvn -S | FileCheck %s

; See PR26774

@X = internal global i32 4

define i32 @test(i32* %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 12, i32* @X
; CHECK-NEXT: call void @doesnotmodX()
; CHECK-NEXT:  %V = load i32, i32* @X
; CHECK-NEXT:  ret i32 %V
  store i32 12, i32* @X
  call void @doesnotmodX( )
  %V = load i32, i32* @X
  ret i32 %V
}

define linkonce_odr void @doesnotmodX() {
  ret void
}

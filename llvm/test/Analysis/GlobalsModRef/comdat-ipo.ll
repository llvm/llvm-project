; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes=gvn -S | FileCheck %s

; See PR26774

@X = internal global i32 4

define i32 @test(ptr %P) {
; CHECK:      @test
; CHECK-NEXT: store i32 12, ptr @X
; CHECK-NEXT: call void @doesnotmodX()
; CHECK-NEXT:  %V = load i32, ptr @X
; CHECK-NEXT:  ret i32 %V
  store i32 12, ptr @X
  call void @doesnotmodX( )
  %V = load i32, ptr @X
  ret i32 %V
}

define linkonce_odr void @doesnotmodX() {
  ret void
}

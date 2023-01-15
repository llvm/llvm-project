; RUN: llvm-link %s %p/Inputs/drop-attribute.ll -S -o - | FileCheck %s

; Test case that checks that nocallback attribute is dropped during linking.

; CHECK: define i32 @main()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @test_nocallback_definition()
; Test that checks that nocallback attribute on a call-site is dropped.
; CHECK-NEXT: call void @test_nocallback_call_site(){{$}}
; CHECK-NEXT: %0 = call float @llvm.sqrt.f32(float undef)
; CHECK-NEXT: call void @test_nocallback_declaration_definition_not_linked_in()
; CHECK-NEXT: call void @test_nocallback_declaration_definition_linked_in()
define i32 @main() {
entry:
  call void @test_nocallback_definition()
  call void @test_nocallback_call_site() nocallback
  call float @llvm.sqrt.f32(float undef)
  call void @test_nocallback_declaration_definition_not_linked_in()
  call void @test_nocallback_declaration_definition_linked_in()
  ret i32 0
}

; Test that checks that nocallback attribute on a definition is dropped.
; CHECK: define void @test_nocallback_definition()
define void @test_nocallback_definition() nocallback {
  ret void
}

; Test that checks that nocallback attribute on a declaration when a definition is linked in is dropped.
; CHECK: declare void @test_nocallback_call_site(){{$}}
declare void @test_nocallback_call_site()

; Test that checks that nocallback attribute on an intrinsic is NOT dropped.
; CHECK: ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
; CHECK-NEXT: declare float @llvm.sqrt.f32(float) #0
declare float @llvm.sqrt.f32(float) nocallback

; Test that checks that nocallback attribute on a declaration when a definition is not linked in is dropped.
; CHECK: declare void @test_nocallback_declaration_definition_not_linked_in(){{$}}
declare void @test_nocallback_declaration_definition_not_linked_in() nocallback

; Test that checks that nocallback attribute on a declaration when a definition is linked in is dropped.
; CHECK: define void @test_nocallback_declaration_definition_linked_in() {{{$}}
declare void @test_nocallback_declaration_definition_linked_in() nocallback

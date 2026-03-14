; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

define void @foo(ptr %ptr, i32 %x) {

  ; CHECK: !mmra metadata attached to unexpected instruction kind
  ; CHECK-NEXT: %bad.add
  %bad.add = add i32 %x, 42, !mmra !{}

  ; CHECK: !mmra metadata attached to unexpected instruction kind
  ; CHECK-NEXT: %bad.sub
  %bad.sub = sub i32 %x, 42, !mmra !{}

  ; CHECK: !mmra metadata attached to unexpected instruction kind
  ; CHECK-NEXT: %bad.sqrt
  %bad.sqrt = call float @llvm.sqrt.f32(float undef), !mmra !{}

  ; CHECK: !mmra expected to be a metadata tuple
  ; CHECK-NEXT: %bad.md0
  ; CHECK-NEXT: !DIFile
  %bad.md0 = load atomic i32, ptr %ptr acquire, align 4, !mmra !0

  ; CHECK: !mmra expected to be a metadata tuple
  ; CHECK-NEXT: %bad.md1
  ; CHECK-NEXT: !DIFile
  %bad.md1 = load atomic i32, ptr %ptr acquire, align 4, !mmra !0

  ; CHECK: !mmra metadata tuple operand is not an MMRA tag
  ; CHECK-NEXT: %bad.md2
  ; CHECK-NEXT: !"foo"
  %bad.md2 = load atomic i32, ptr %ptr acquire, align 4, !mmra !1

  ; CHECK: !mmra metadata tuple operand is not an MMRA tag
  ; CHECK-NEXT: %bad.md3
  ; CHECK-NEXT: !"baz"
  %bad.md3 = load atomic i32, ptr %ptr acquire, align 4, !mmra !2
  ret void
}

declare float @llvm.sqrt.f32(float)

!0 = !DIFile(filename: "test.c", directory: "")
!1 = !{!"foo", !"bar", !"bux"}
!2 = !{!"baz", !0}

; RUN: llc -mtriple=arm64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s

@a = internal unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [6 x i8] c"test\0A\00", align 1

define i32 @get() {
; CHECK: get:
; CHECK: adrp x{{[0-9]+}}, a
; CHECK-NEXT: ldr w{{[0-9]+}}, [x{{[0-9]}}, :lo12:a]
  %res = load i32, ptr @a, align 4
  ret i32 %res
}

define void @foo() nounwind {
; CHECK: foo:
; CHECK: adrp x{{[0-9]}}, .L.str
; CHECK-NEXT: add x{{[0-9]}}, x{{[0-9]}}, :lo12:.L.str
  tail call void @bar(ptr @.str)
  ret void
}

declare void @bar(ptr)

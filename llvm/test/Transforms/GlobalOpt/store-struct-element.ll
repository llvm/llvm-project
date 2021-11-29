; RUN: opt < %s -passes=globalopt -S -o - | FileCheck %s

%class.Class = type { i8, i8, i8, i8 }
@A = local_unnamed_addr global %class.Class undef, align 4
@B = local_unnamed_addr global %class.Class undef, align 4

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 65535, void ()* @initA, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* @initB, i8* null }
]

define internal void @initA() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  store i32 -1, i32* bitcast (%class.Class* @A to i32*), align 4
  ret void
}

define internal void @initB() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  store i8 -1, i8* bitcast (%class.Class* @B to i8*), align 4
  ret void
}

; rdar://79503568
; Check that we don't miscompile when the store covers the whole struct.
; CHECK-NOT: @A = local_unnamed_addr global %class.Class { i8 -1, i8 undef, i8 undef, i8 undef }, align 4

; FIXME: We could optimzie this as { i8 -1, i8 -1, i8 -1, i8 -1 } if constant folding were a little smarter.
; CHECK: @A = local_unnamed_addr global %class.Class undef, align 4

; Check that we still perform the transform when store is smaller than the width of the 0th element.
; CHECK: @B = local_unnamed_addr global %class.Class { i8 -1, i8 undef, i8 undef, i8 undef }, align 4

; CHECK: define internal void @initA()
; CHECK-NOT: define internal void @initB()


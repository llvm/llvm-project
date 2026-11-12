; Ensure that we don't emit available_externally functions at -O2, unless
; -flto is present in which case we should preserve them for link-time inlining
; decisions.
; RUN: opt < %s -S -passes='default<O2>' | FileCheck %s
; RUN: opt < %s -S -passes='lto-pre-link<O2>' | FileCheck %s --check-prefix=LTO

@x = common local_unnamed_addr global i32 0, align 4

define void @test() local_unnamed_addr #0 {
entry:
  tail call void @f0(i32 17)
  ret void
}

; CHECK: declare void @f0(i32)
; LTO: define available_externally void @f0(i32 %y)
define available_externally void @f0(i32 %y) local_unnamed_addr #0 {
entry:
  store i32 %y, ptr @x, align 4
  ret void
}

attributes #0 = { noinline }

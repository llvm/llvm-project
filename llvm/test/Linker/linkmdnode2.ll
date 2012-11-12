; This file is used by linkmdnode.ll, so it doesn't actually do anything itself
;
; RUN: true

!22 = metadata !{i32 42, metadata !"foobar"}

declare i8 @llvm.something(metadata %a)
define void @foo1() {
  ;; Intrinsic using MDNode and MDString
  %x = call i8 @llvm.something(metadata !22)
  ret void
}



; PR9015
define void @test() {
  ret void, !abc !0
}

!0 = metadata !{metadata !0, i32 42 }


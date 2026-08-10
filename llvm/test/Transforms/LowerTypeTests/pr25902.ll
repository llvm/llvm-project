; PR25902: gold plugin crash.
; RUN: opt -mtriple=i686-pc -S -passes=lowertypetests < %s

define void @f(ptr %p) {
entry:
  %b = call i1 @llvm.type.test(ptr %p, metadata !"_ZTSFvvE"), !nosanitize !1
  ret void
}

define void @g() !type !0 {
entry:
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)

!0 = !{i64 0, !"_ZTSFvvE"}
!1 = !{}

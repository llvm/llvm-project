; PR25902: gold plugin crash.
; RUN: opt -mtriple=i686-pc -S -lowertypetests < %s

define void @f(void ()* %p) {
entry:
  %a = bitcast void ()* %p to i8*, !nosanitize !1
  %b = call i1 @llvm.type.test(i8* %a, metadata !"_ZTSFvvE"), !nosanitize !1
  ret void
}

define void @g() !type !0 {
entry:
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)

!0 = !{i64 0, !"_ZTSFvvE"}
!1 = !{}

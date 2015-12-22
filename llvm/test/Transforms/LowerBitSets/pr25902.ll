; PR25902: gold plugin crash.
; RUN: opt -mtriple=i686-pc -S -lowerbitsets < %s

define void @f(void ()* %p) {
entry:
  %a = bitcast void ()* %p to i8*, !nosanitize !1
  %b = call i1 @llvm.bitset.test(i8* %a, metadata !"_ZTSFvvE"), !nosanitize !1
  ret void
}

define void @g() {
entry:
  ret void
}

declare i1 @llvm.bitset.test(i8*, metadata)

!llvm.bitsets = !{!0}

!0 = !{!"_ZTSFvvE", void ()* @g, i64 0}
!1 = !{}

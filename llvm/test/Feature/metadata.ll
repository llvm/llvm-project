; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR7105

define void @foo(i32 %x) {
  call void @llvm.zonk(metadata !1, i64 0, metadata !1)
  store i32 0, i32* null, !whatever !0, !whatever_else !{}, !more !{metadata !"hello"}
  store i32 0, i32* null, !whatever !{i32 %x, metadata !"hello", metadata !1, metadata !{}, metadata !2}
  ret void, !whatever !{i32 %x}
}

declare void @llvm.zonk(metadata, i64, metadata) nounwind readnone

!named = !{!0}
!another_named = !{}
!0 = metadata !{i8** null}
!1 = metadata !{i8* null, metadata !2}
!2 = metadata !{}

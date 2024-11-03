; RUN: not --crash opt -passes=lowertypetests %s -disable-output
target datalayout = "e-p:64:64:64:32"

@a = constant i32 1, !type !0
@b = constant [2 x i32] [i32 2, i32 3], !type !1

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"typeid1"}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

; CHECK: @bar(
define i1 @bar() {
  ; CHECK: ret i1 true
  %x = call i1 @llvm.type.test(ptr getelementptr ([2 x i32], ptr @b, i32 0, i32 1), metadata !"typeid1")
  ret i1 %x
}

; RUN: llc < %s -relocation-model=pic -mtriple=x86_64-pc-linux | FileCheck %s
; RUN: llc < %s -relocation-model=pic -mtriple=i386-pc-linux | FileCheck --check-prefix=I386 %s

@foo1 = extern_weak hidden global i32, align 4
define i32* @bar1() {
  ret i32* @foo1
}
; CHECK: bar1:
; CHECK: movq foo1@GOTPCREL(%rip), %rax
; I386: bar1:
; I386: movl foo1@GOT(%eax), %eax

@foo2 = external hidden global i32, align 4
define i32* @bar2() {
  ret i32* @foo2
}
; CHECK: bar2:
; CHECK: leaq foo2(%rip), %rax
; I386: bar2:
; I386: leal foo2@GOTOFF(%eax), %eax

declare extern_weak hidden void @foo3()
define void @bar3() {
  call void @foo3()
  ret void
}
; CHECK: bar3:
; CHECK: callq foo3
; I386: bar3:
; I386: calll foo3

declare external hidden void @foo4()
define void @bar4() {
  call void @foo4()
  ret void
}
; CHECK: bar4:
; CHECK: callq foo4
; I386: bar4:
; I386: calll foo4

declare extern_weak hidden i32 @foo5()
define i32()* @bar5() {
  ret i32()* @foo5
}
; CHECK: bar5:
; CHECK: movq foo5@GOTPCREL(%rip), %rax
; I386: bar5:
; I386: movl foo5@GOT(%eax), %eax

declare external hidden i32 @foo6()
define i32()* @bar6() {
  ret i32()* @foo6
}
; CHECK: bar6:
; CHECK: leaq foo6(%rip), %rax
; I386: bar6:
; I386: leal foo6@GOTOFF(%eax), %eax

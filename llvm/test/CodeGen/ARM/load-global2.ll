; PR35221. Test that external global address is not reloaded from GOT in each BB.
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=LINUX-PIC

@x = external global i8, align 1

define signext i8 @foo() {
entry:
; LINUX-PIC:     ldr	r[[A:.]], .LCPI0_0
; LINUX-PIC:     ldr	r[[B:.]], [pc, r[[A]]]
; LINUX-PIC:     ldrb	r{{.}}, [r[[B]]]
  %0 = load i8, i8* @x
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %bb1, label %bb2

bb1:
  call void @bar()
; No more pc-relative loads! Reuse r[[B]].
; LINUX-PIC:     bl	bar
; LINUX-PIC-NOT: ldr{{.*}}[pc,
; LINUX-PIC:     ldrsb	r{{.}}, [r[[B]]]
  %1 = load i8, i8* @x
  ret i8 %1

bb2:
  ret i8 0
}

declare void @bar()



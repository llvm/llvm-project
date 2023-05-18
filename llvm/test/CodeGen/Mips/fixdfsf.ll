; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=pic1
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=pic2

@x = common global double 0.000000e+00, align 8
@y = common global i32 0, align 4

; Function Attrs: nounwind optsize
define void @foo()  {
entry:
  %0 = load double, ptr @x, align 8
  %conv = fptoui double %0 to i32
  store i32 %conv, ptr @y, align 4
; pic1:	lw	${{[0-9]+}}, %call16(__fixunsdfsi)(${{[0-9]+}})
; pic2:	lw	${{[0-9]+}}, %got(__mips16_call_stub_2)(${{[0-9]+}})
  ret void
}



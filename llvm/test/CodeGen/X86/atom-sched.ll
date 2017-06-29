; RUN: llc <%s -O2 -mcpu=atom -march=x86 -relocation-model=static | FileCheck -check-prefix=atom %s
; RUN: llc <%s -O2 -mcpu=slm -march=x86 -relocation-model=static | FileCheck -check-prefix=slm %s
; RUN: llc <%s -O2 -mcpu=goldmont -march=x86 -relocation-model=static | FileCheck -check-prefix=slm %s
; RUN: llc <%s -O2 -mcpu=core2 -march=x86 -relocation-model=static | FileCheck %s
;

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4
@d = common global i32 0, align 4
@e = common global i32 0, align 4
@f = common global i32 0, align 4

define void @func() nounwind uwtable {
; atom: imull
; atom-NOT: movl
; atom: imull
; slm: imull
; slm-NOT: movl
; slm: imull
; CHECK: imull
; CHECK: movl
; CHECK: imull
entry:
  %0 = load i32, i32* @b, align 4
  %1 = load i32, i32* @c, align 4
  %mul = mul nsw i32 %0, %1
  store i32 %mul, i32* @a, align 4
  %2 = load i32, i32* @e, align 4
  %3 = load i32, i32* @f, align 4
  %mul1 = mul nsw i32 %2, %3
  store i32 %mul1, i32* @d, align 4
  ret void
}

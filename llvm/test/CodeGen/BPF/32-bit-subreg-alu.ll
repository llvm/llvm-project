; RUN: llc -O2 -march=bpfel -mattr=+alu32 < %s | FileCheck %s
; RUN: llc -O2 -march=bpfel -mcpu=v3 < %s | FileCheck %s
;
; int mov(int a)
; {
;   return a;
; }
;
; int mov_ri(void)
; {
;   return 0xff;
; }
;
; int add(int a, int b)
; {
;   return a + b;
; }
;
; int add_i(int a)
; {
;   return a + 0x7fffffff;
; }
;
; int sub(int a, int b)
; {
;   return a - b;
; }
;
; int sub_i(int a)
; {
;   return a - 0xffffffff;
; }
;
; int mul(int a, int b)
; {
;   return a * b;
; }
;
; int mul_i(int a)
; {
;   return a * 0xf;
; }
;
; unsigned div(unsigned a, unsigned b)
; {
;   return a / b;
; }
;
; unsigned div_i(unsigned a)
; {
;   return a / 0xf;
; }
;
; unsigned rem(unsigned a, unsigned b)
; {
;   return a % b;
; }
;
; unsigned rem_i(unsigned a)
; {
;   return a % 0xf;
; }
;
; int or(int a, int b)
; {
;   return a | b;
; }
;
; int or_i(int a)
; {
;   return a | 0xff;
; }
;
; int xor(int a, int b)
; {
;   return a ^ b;
; }
;
; int xor_i(int a)
; {
;   return a ^ 0xfff;
; }
;
; int and(int a, int b)
; {
;   return a & b;
; }
;
; int and_i(int a)
; {
;   return a & 0xffff;
; }
;
; int sll(int a, int b)
; {
;   return a << b;
; }
;
; int sll_i(int a)
; {
;   return a << 17;
; }
;
; unsigned srl(unsigned a, unsigned b)
; {
;   return a >> b;
; }
;
; unsigned srl_i(unsigned a, unsigned b)
; {
;   return a >> 31;
; }
;
; int sra(int a, int b)
; {
;   return a >> b;
; }
;
; int sra_i(int a, int b)
; {
;   return a >> 7;
; }
;
; int neg(int a)
; {
;   return -a;
; }

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @mov(i32 returned %a) local_unnamed_addr #0 {
entry:
  ret i32 %a
; CHECK: w{{[0-9]+}} = w{{[0-9]+}}
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @mov_ri() local_unnamed_addr #0 {
entry:
  ret i32 255
; CHECK: w{{[0-9]+}} = 255
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @add(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %b, %a
; CHECK: w{{[0-9]+}} += w{{[0-9]+}}
  ret i32 %add
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @add_i(i32 %a) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %a, 2147483647
; CHECK: w{{[0-9]+}} += 2147483647
  ret i32 %add
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sub(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i32 %a, %b
; CHECK: w{{[0-9]+}} -= w{{[0-9]+}}
  ret i32 %sub
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sub_i(i32 %a) local_unnamed_addr #0 {
entry:
  %sub = add i32 %a, 1
; CHECK: w{{[0-9]+}} += 1
  ret i32 %sub
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @mul(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %mul = mul nsw i32 %b, %a
; CHECK: w{{[0-9]+}} *= w{{[0-9]+}}
  ret i32 %mul
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @mul_i(i32 %a) local_unnamed_addr #0 {
entry:
  %mul = mul nsw i32 %a, 15
; CHECK: w{{[0-9]+}} *= 15
  ret i32 %mul
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @div(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %div = udiv i32 %a, %b
; CHECK: w{{[0-9]+}} /= w{{[0-9]+}}
  ret i32 %div
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @div_i(i32 %a) local_unnamed_addr #0 {
entry:
  %div = udiv i32 %a, 15
; CHECK: w{{[0-9]+}} /= 15
  ret i32 %div
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @rem(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %rem = urem i32 %a, %b
; CHECK: w{{[0-9]+}} %= w{{[0-9]+}}
  ret i32 %rem
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @rem_i(i32 %a) local_unnamed_addr #0 {
entry:
  %rem = urem i32 %a, 15
; CHECK: w{{[0-9]+}} %= 15
  ret i32 %rem
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @or(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %or = or i32 %b, %a
; CHECK: w{{[0-9]+}} |= w{{[0-9]+}}
  ret i32 %or
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @or_i(i32 %a) local_unnamed_addr #0 {
entry:
  %or = or i32 %a, 255
; CHECK: w{{[0-9]+}} |= 255
  ret i32 %or
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @xor(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %xor = xor i32 %b, %a
; CHECK: w{{[0-9]+}} ^= w{{[0-9]+}}
  ret i32 %xor
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @xor_i(i32 %a) local_unnamed_addr #0 {
entry:
  %xor = xor i32 %a, 4095
; CHECK: w{{[0-9]+}} ^= 4095
  ret i32 %xor
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @and(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %and = and i32 %b, %a
; CHECK: w{{[0-9]+}} &= w{{[0-9]+}}
  ret i32 %and
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @and_i(i32 %a) local_unnamed_addr #0 {
entry:
  %and = and i32 %a, 65535
; CHECK: w{{[0-9]+}} &= 65535
  ret i32 %and
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sll(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, %b
; CHECK: w{{[0-9]+}} <<= w{{[0-9]+}}
  ret i32 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sll_i(i32 %a) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 17
; CHECK: w{{[0-9]+}} <<= 17
  ret i32 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @srl(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %shr = lshr i32 %a, %b
; CHECK: w{{[0-9]+}} >>= w{{[0-9]+}}
  ret i32 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @srl_i(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %shr = lshr i32 %a, 31
; CHECK: w{{[0-9]+}} >>= 31
  ret i32 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sra(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %shr = ashr i32 %a, %b
; CHECK: w{{[0-9]+}} s>>= w{{[0-9]+}}
  ret i32 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @sra_i(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %shr = ashr i32 %a, 7
; CHECK: w{{[0-9]+}} s>>= 7
  ret i32 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @neg(i32 %a) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i32 0, %a
; CHECK: w{{[0-9]+}} = -w{{[0-9]+}}
  ret i32 %sub
}

; RUN: llc < %s -mtriple thumbv7 | FileCheck %s

; ModuleID = 'bugpoint-reduced-simplified.bc'
define hidden void @bn_mul_comba8(ptr nocapture %r, ptr nocapture readonly %a, ptr nocapture readonly %b) local_unnamed_addr {
entry:
; This test is actually checking that no cycle is introduced but at least we
; want to see a couple of umull and one umlal in the output
; CHECK: umull
; CHECK: umull
; CHECK: umlal
  %0 = load i32, ptr %a, align 4
  %conv = zext i32 %0 to i64
  %1 = load i32, ptr %b, align 4
  %conv2 = zext i32 %1 to i64
  %mul = mul nuw i64 %conv2, %conv
  %shr = lshr i64 %mul, 32
  %2 = load i32, ptr %a, align 4
  %conv13 = zext i32 %2 to i64
  %3 = load i32, ptr undef, align 4
  %conv15 = zext i32 %3 to i64
  %mul16 = mul nuw i64 %conv15, %conv13
  %add18 = add i64 %mul16, %shr
  %shr20 = lshr i64 %add18, 32
  %conv21 = trunc i64 %shr20 to i32
  %4 = load i32, ptr undef, align 4
  %conv34 = zext i32 %4 to i64
  %5 = load i32, ptr %b, align 4
  %conv36 = zext i32 %5 to i64
  %mul37 = mul nuw i64 %conv36, %conv34
  %conv38 = and i64 %add18, 4294967295
  %add39 = add i64 %mul37, %conv38
  %shr41 = lshr i64 %add39, 32
  %conv42 = trunc i64 %shr41 to i32
  %add43 = add i32 %conv42, %conv21
  %cmp44 = icmp ult i32 %add43, %conv42
  %c1.1 = zext i1 %cmp44 to i32
  %add65 = add i32 0, %c1.1
  %add86 = add i32 %add65, 0
  %add107 = add i32 %add86, 0
  %conv124 = zext i32 %add107 to i64
  %add125 = add i64 0, %conv124
  %conv145 = and i64 %add125, 4294967295
  %add146 = add i64 %conv145, 0
  %conv166 = and i64 %add146, 4294967295
  %add167 = add i64 %conv166, 0
  %conv187 = and i64 %add167, 4294967295
  %add188 = add i64 %conv187, 0
  %conv189 = trunc i64 %add188 to i32
  %arrayidx200 = getelementptr inbounds i32, ptr %r, i32 3
  store i32 %conv189, ptr %arrayidx200, align 4
  ret void
}


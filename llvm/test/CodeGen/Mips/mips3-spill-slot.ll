; RUN: llc -march=mips64 -mcpu=mips3 < %s 2>&1 | FileCheck %s --check-prefix=CHECK
; This testcase is from PR35859.
; Check that spill slot has the correct size for mips3 and n64 ABI.
; Previously, this test case would fail during register
; scavenging due to wrong spill slot size.

; CHECK-NOT: Cannot scavenge register without an emergency spill slot!

@n = external local_unnamed_addr global i32*, align 8

define void @o(i32* nocapture readonly %a, i64* %b) local_unnamed_addr {
entry:
  %0 = load i32, i32* undef, align 4
  %and12 = and i32 %0, 67295
  %1 = zext i32 %and12 to i64
  %conv16 = sext i32 %0 to i64
  %2 = ptrtoint i64* %b to i64
  %mul22 = mul nsw i64 %1, %2
  %mul23 = mul nsw i64 %conv16, %2
  %tobool25 = icmp ne i64 %mul22, 0
  %inc27 = zext i1 %tobool25 to i64
  %3 = load i32*, i32** @n, align 8
  %arrayidx36 = getelementptr inbounds i32, i32* %3, i64 4
  store i32 0, i32* %arrayidx36, align 4
  %spec.select = add i64 0, %mul23
  %hi14.0 = add i64 %spec.select, %inc27
  %add51 = add i64 %hi14.0, 0
  %4 = load i32, i32* null, align 4
  %and59 = and i32 %4, 67295
  %5 = zext i32 %and59 to i64
  %conv63 = sext i32 %4 to i64
  %6 = load i64, i64* %b, align 8
  %mul71 = mul nsw i64 %6, %5
  %mul72 = mul nsw i64 %6, %conv63
  %tobool74 = icmp ne i64 %mul71, 0
  %inc76 = zext i1 %tobool74 to i64
  %arrayidx85 = getelementptr inbounds i32, i32* %a, i64 5
  %7 = load i32, i32* %arrayidx85, align 4
  %and86 = and i32 %7, 67295
  %conv90 = sext i32 %7 to i64
  %8 = load i64, i64* undef, align 8
  %mul99 = mul nsw i64 %8, %conv90
  %9 = load i32, i32* undef, align 4
  %and113 = and i32 %9, 67295
  %tobool126 = icmp eq i32 %and113, 0
  %spec.select397.v = select i1 %tobool126, i64 2, i64 3
  %10 = load i32, i32* undef, align 4
  %and138 = and i32 %10, 67295
  %11 = zext i32 %and138 to i64
  %conv142 = sext i32 %10 to i64
  %12 = load i64, i64* null, align 8
  %mul150 = mul nsw i64 %12, %11
  %mul151 = mul nsw i64 %12, %conv142
  %tobool153 = icmp ne i64 %mul150, 0
  %inc155 = zext i1 %tobool153 to i64
  %add157 = add nsw i64 0, %11
  %spec.select398 = add i64 0, %mul72
  %hi140.0 = add i64 %spec.select398, %8
  %spec.select397 = add i64 %hi140.0, %mul99
  %hi115.0 = add i64 %spec.select397, %inc76
  %add100 = add i64 %hi115.0, 0
  %spec.select396 = add i64 %add100, %12
  %hi88.0 = add i64 %spec.select396, 0
  %add73 = add i64 %hi88.0, %spec.select397.v
  %spec.select395 = add i64 %add73, %mul151
  %hi61.0 = add i64 %spec.select395, %inc155
  %add83 = add i64 %hi61.0, 0
  %add110 = add i64 %add83, 0
  %add135 = add i64 %add110, 0
  %add162 = add i64 %add135, 0
  %13 = load i32, i32* null, align 4
  %and165 = and i32 %13, 67295
  %14 = zext i32 %and165 to i64
  %conv169 = sext i32 %13 to i64
  %mul175 = mul nsw i64 %14, %2
  %mul176 = mul nsw i64 %conv169, %2
  %tobool178 = icmp ne i64 %mul175, 0
  %inc180 = zext i1 %tobool178 to i64
  %add182 = sub nsw i64 0, %14
  %tobool183 = icmp ne i64 %add162, %add182
  %inc185 = zext i1 %tobool183 to i64
  %add177 = add i64 %add51, %2
  %spec.select399 = add i64 %add177, %mul176
  %hi167.0 = add i64 %spec.select399, %inc180
  %add187 = add i64 %hi167.0, %inc185
  %tobool203 = icmp eq i32 %and86, 0
  %spec.select400.v = select i1 %tobool203, i64 3, i64 4
  %spec.select400 = add nsw i64 %spec.select400.v, 0
  %tobool208 = icmp ne i64 %add187, 0
  %inc210 = zext i1 %tobool208 to i64
  %hi192.0 = add i64 %spec.select400, %add157
  %add212 = add i64 %hi192.0, %inc210
  %15 = inttoptr i64 %add212 to i32*
  store i32* %15, i32** @n, align 8
  ret void
}

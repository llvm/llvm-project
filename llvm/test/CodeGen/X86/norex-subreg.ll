; RUN: llc -O0 < %s -verify-machineinstrs
; RUN: llc < %s -verify-machineinstrs
target triple = "x86_64-apple-macosx10.7"

; This test case extracts a sub_8bit_hi sub-register:
;
;	%r8b<def> = COPY %bh, %ebx<imp-use,kill>
;	%esi<def> = MOVZX32_NOREXrr8 %r8b<kill>
;
; The register allocation above is invalid, %bh can only be encoded without an
; REX prefix, so the destination register must be GR8_NOREX.  The code above
; triggers an assertion in copyPhysReg.
;
; <rdar://problem/10248099>

define void @f() nounwind uwtable ssp {
entry:
  %0 = load i32, i32* undef, align 4
  %add = add i32 0, %0
  %conv1 = trunc i32 %add to i16
  %bf.value = and i16 %conv1, 255
  %1 = and i16 %bf.value, 255
  %2 = shl i16 %1, 8
  %3 = load i16, i16* undef, align 1
  %4 = and i16 %3, 255
  %5 = or i16 %4, %2
  store i16 %5, i16* undef, align 1
  %6 = load i16, i16* undef, align 1
  %7 = lshr i16 %6, 8
  %bf.clear2 = and i16 %7, 255
  %conv3 = zext i16 %bf.clear2 to i32
  %rem = srem i32 %conv3, 15
  %conv4 = trunc i32 %rem to i16
  %bf.value5 = and i16 %conv4, 255
  %8 = and i16 %bf.value5, 255
  %9 = shl i16 %8, 8
  %10 = or i16 undef, %9
  store i16 %10, i16* undef, align 1
  ret void
}

; This test case extracts a sub_8bit_hi sub-register:
;
;       %2<def> = COPY %1:sub_8bit_hi; GR8:%2 GR64_ABCD:%1
;       TEST8ri %2, 1, %eflags<imp-def>; GR8:%2
;
; %2 must be constrained to GR8_NOREX, or the COPY could become impossible.
;
; PR11088

define fastcc i32 @g(i64 %FB) nounwind uwtable readnone align 2 {
entry:
  %and32 = and i64 %FB, 256
  %cmp33 = icmp eq i64 %and32, 0
  %Features.6.or35 = select i1 %cmp33, i32 0, i32 undef
  %cmp38 = icmp eq i64 undef, 0
  %or40 = or i32 %Features.6.or35, 4
  %Features.8 = select i1 %cmp38, i32 %Features.6.or35, i32 %or40
  %and42 = and i64 %FB, 32
  %or45 = or i32 %Features.8, 2
  %cmp43 = icmp eq i64 %and42, 0
  %Features.8.or45 = select i1 %cmp43, i32 %Features.8, i32 %or45
  %and47 = and i64 %FB, 8192
  %cmp48 = icmp eq i64 %and47, 0
  %or50 = or i32 %Features.8.or45, 32
  %Features.10 = select i1 %cmp48, i32 %Features.8.or45, i32 %or50
  %or55 = or i32 %Features.10, 64
  %Features.10.or55 = select i1 undef, i32 %Features.10, i32 %or55
  %and57 = lshr i64 %FB, 2
  %and57.tr = trunc i64 %and57 to i32
  %or60 = and i32 %and57.tr, 1
  %Features.12 = or i32 %Features.10.or55, %or60
  %and62 = and i64 %FB, 128
  %or65 = or i32 %Features.12, 8
  %cmp63 = icmp eq i64 %and62, 0
  %Features.12.or65 = select i1 %cmp63, i32 %Features.12, i32 %or65
  %Features.14 = select i1 undef, i32 undef, i32 %Features.12.or65
  %Features.16 = select i1 undef, i32 undef, i32 %Features.14
  ret i32 %Features.16
}

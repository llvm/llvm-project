; We run the tests with both the default optimization level and O0, to make sure
; we don't have any ABI differences between them. In principle, the ABI checks
; should be the same for both optimization levels (there could be exceptions
; from this when a div and a mod with the same operands are not coallesced into
; the same divmod, but luckily this doesn't occur in practice even at O0).
; Sometimes the checks that the correct registers are used after the libcalls
; are different between optimization levels, so we have to separate them.
; RUN: llc -mtriple armv7-none-eabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-none-eabi %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-none-eabihf %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-none-eabihf %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefix=EABI
; All "eabi" (Bare, GNU and Android) must lower SREM/UREM to __aeabi_{u,i}divmod
; RUN: llc -mtriple armv7-linux-androideabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-androideabi %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-gnueabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-gnueabi %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-musleabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-musleabi %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-apple-darwin %s -o - | FileCheck %s --check-prefixes=DARWIN,DARWIN-DEFAULT
; RUN: llc -mtriple armv7-apple-darwin %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefixes=DARWIN,DARWIN-O0
; FIXME: long-term, we will use "-apple-macho" and won't need this exception:
; RUN: llc -mtriple armv7-apple-darwin-eabi %s -o - | FileCheck %s --check-prefixes=DARWIN,DARWIN-DEFAULT
; RUN: llc -mtriple armv7-apple-darwin-eabi %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefixes=DARWIN,DARWIN-O0
; RUN: llc -mtriple thumbv7-windows %s -o - | FileCheck %s --check-prefixes=WINDOWS,WINDOWS-DEFAULT
; RUN: llc -mtriple thumbv7-windows %s -o - -O0 -optimize-regalloc | FileCheck %s --check-prefixes=WINDOWS,WINDOWS-O0

define signext i16 @f16(i16 signext %a, i16 signext %b) {
; EABI-LABEL: f16:
; DARWIN-LABEL: f16:
; WINDOWS-LABEL: f16:
entry:
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %div = sdiv i32 %conv, %conv1
  %rem = srem i32 %conv, %conv1
; EABI: __aeabi_idivmod
; EABI: mov [[div:r[0-9]+]], r0
; EABI: mov [[rem:r[0-9]+]], r1
; DARWIN: ___divsi3
; DARWIN: mov [[div:r[0-9]+]], r0
; DARWIN: __modsi3
; DARWIN-DEFAULT: add [[sum:r[0-9]+]], r0, [[div]]
; DARWIN-O0: mov [[rem:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS-DEFAULT: mls [[rem:r[0-9]+]], r0,
; WINDOWS-DEFAULT: adds [[sum:r[0-9]+]], [[rem]], r0
; WINDOWS-O0: mov [[div:r[0-9]+]], r0
; WINDOWS-O0: mls [[rem:r[0-9]+]], [[div]],
  %rem8 = srem i32 %conv1, %conv
; EABI: __aeabi_idivmod
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem1:r[0-9]+]], r0,
  %add = add nsw i32 %rem, %div
  %add13 = add nsw i32 %add, %rem8
  %conv14 = trunc i32 %add13 to i16
; EABI: add r0{{.*}}r1
; EABI: sxth r0, r0
; DARWIN-DEFAULT: add [[res:r[0-9]+]], [[sum]], r0
; DARWIN-O0: add [[sum:r[0-9]+]], [[rem]], [[div]]
; DARWIN-O0: add [[res:r[0-9]+]], [[sum]], r0
; DARWIN: sxth r0, [[res]]
; WINDOWS-O0: adds [[sum:r[0-9]+]], [[rem]], [[div]]
; WINDOWS: add [[rem1]], [[sum]]
; WINDOWS: sxth [[res:r[0-9]+]], [[rem1]]
  ret i16 %conv14
}

define i32 @f32(i32 %a, i32 %b) {
; EABI-LABEL: f32:
; DARWIN-LABEL: f32:
; WINDOWS-LABEL: f32:
entry:
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; EABI: mov [[div:r[0-9]+]], r0
; EABI: mov [[rem:r[0-9]+]], r1
; DARWIN: ___divsi3
; DARWIN: mov [[div:r[0-9]+]], r0
; DARWIN: __modsi3
; DARWIN-DEFAULT: add [[sum:r[0-9]+]], r0, [[div]]
; DARWIN-O0: mov [[rem:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mov [[div:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
; WINDOWS-DEFAULT: add [[div]], [[rem]]
  %rem1 = srem i32 %b, %a
; EABI: __aeabi_idivmod
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem1:r[0-9]+]], r0,
  %add = add nsw i32 %rem, %div
  %add2 = add nsw i32 %add, %rem1
; EABI: add r0{{.*}}r1
; DARWIN-DEFAULT: add r0, [[sum]], r0
; DARWIN-O0: add [[sum:r[0-9]+]], [[rem]], [[div]]
; DARWIN-O0: add [[res:r[0-9]+]], [[sum]], r0
; WINDOWS-DEFAULT: add [[rem1]], [[div]]
; WINDOWS-O0: adds [[sum:r[0-9]+]], [[rem]], [[div]]
; WINDOWS-O0: add [[rem1]], [[sum]]
  ret i32 %add2
}

define i32 @uf(i32 %a, i32 %b) {
; EABI-LABEL: uf:
; DARWIN-LABEL: uf:
; WINDOWS-LABEL: uf:
entry:
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
; EABI: __aeabi_uidivmod
; DARWIN: ___udivsi3
; DARWIN: mov [[div:r[0-9]+]], r0
; DARWIN: __umodsi3
; DARWIN-DEFAULT: add [[sum:r[0-9]+]], r0, [[div]]
; DARWIN-O0: mov [[rem:r[0-9]+]], r0
; WINDOWS: __rt_udiv
; WINDOWS: mov [[div:r[0-9]+]], r0
; WINDOWS: __rt_udiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
; WINDOWS-DEFAULT: add [[div]], [[rem]]
  %rem1 = urem i32 %b, %a
; EABI: __aeabi_uidivmod
; DARWIN: __umodsi3
; WINDOWS: __rt_udiv
; WINDOWS: mls [[rem1:r[0-9]+]], r0,
  %add = add nuw i32 %rem, %div
  %add2 = add nuw i32 %add, %rem1
; EABI: add r0{{.*}}r1
; DARWIN-DEFAULT: add r0, [[sum]], r0
; DARWIN-O0: add [[sum:r[0-9]+]], [[rem]], [[div]]
; DARWIN-O0: add [[res:r[0-9]+]], [[sum]], r0
; WINDOWS-DEFAULT: add [[rem1]], [[div]]
; WINDOWS-O0: adds [[sum:r[0-9]+]], [[rem]], [[div]]
; WINDOWS-O0: add [[rem1]], [[sum]]
  ret i32 %add2
}

define i64 @longf(i64 %a, i64 %b) {
; EABI-LABEL: longf:
; DARWIN-LABEL: longf:
; WINDOWS-LABEL: longf:
entry:
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %b
; EABI: __aeabi_ldivmod
; EABI-NEXT: adds r0
; EABI-NEXT: adc r1
; EABI-NOT: __aeabi_ldivmod
; DARWIN: ___divdi3
; DARWIN: mov [[div1:r[0-9]+]], r0
; DARWIN: mov [[div2:r[0-9]+]], r1
; DARWIN: __moddi3
; WINDOWS: __rt_sdiv64
; WINDOWS: mov [[div1:r[0-9]+]], r0
; WINDOWS: mov [[div2:r[0-9]+]], r1
; WINDOWS: __moddi3
  %add = add nsw i64 %rem, %div
; DARWIN: adds r0{{.*}}[[div1]]
; DARWIN: adc r1{{.*}}[[div2]]
; WINDOWS: adds.w r0, r0, [[div1]]
; WINDOWS: adc.w r1, r1, [[div2]]
  ret i64 %add
}

define i16 @shortf(i16 %a, i16 %b) {
; EABI-LABEL: shortf:
; DARWIN-LABEL: shortf:
; WINDOWS-LABEL: shortf:
entry:
  %div = sdiv i16 %a, %b
  %rem = srem i16 %a, %b
; EABI: __aeabi_idivmod
; DARWIN: ___divsi3
; DARWIN: mov [[div1:r[0-9]+]], r0
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mov [[div:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
  %add = add nsw i16 %rem, %div
; EABI: add r0, r1
; DARWIN: add r0{{.*}}[[div1]]
; WINDOWS: add [[rem]], [[div]]
  ret i16 %add
}

define i32 @g1(i32 %a, i32 %b) {
; EABI-LABEL: g1:
; DARWIN-LABEL: g1:
; WINDOWS-LABEL: g1:
entry:
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mov [[div:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
  %add = add nsw i32 %rem, %div
; EABI:	add	r0{{.*}}r1
; DARWIN: add r0{{.*}}[[sum]]
; WINDOWS: add [[rem]], [[div]]
  ret i32 %add
}

; On both Darwin and Gnu, this is just a call to __modsi3
define i32 @g2(i32 %a, i32 %b) {
; EABI-LABEL: g2:
; DARWIN-LABEL: g2:
; WINDOWS-LABEL: g2:
entry:
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
  ret i32 %rem
; EABI:	mov	r0, r1
; WINDOWS: mls r0, r0,
}

define i32 @g3(i32 %a, i32 %b) {
; EABI-LABEL: g3:
; DARWIN-LABEL: g3:
; WINDOWS-LABEL: g3:
entry:
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; EABI: mov [[mod:r[0-9]+]], r1
; DARWIN: __modsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
  %rem1 = srem i32 %b, %rem
; EABI: __aeabi_idivmod
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem1:r[0-9]+]], r0,
  %add = add nsw i32 %rem1, %rem
; EABI: add r0, r1, [[mod]]
; DARWIN: add r0{{.*}}[[sum]]
; WINDOWS: add [[rem1]], [[rem]]
  ret i32 %add
}

define i32 @g4(i32 %a, i32 %b) {
; EABI-LABEL: g4:
; DARWIN-LABEL: g4:
; WINDOWS-LABEL: g4:
entry:
  %div = sdiv i32 %a, %b
; EABI: __aeabi_idiv{{$}}
; EABI: mov [[div:r[0-9]+]], r0
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; WINDOWS: __rt_sdiv
; WINDOWS: mov [[div:r[0-9]+]], r0
  %rem = srem i32 %b, %div
; EABI: __aeabi_idivmod
; DARWIN: __modsi3
; WINDOWS: __rt_sdiv
; WINDOWS: mls [[rem:r[0-9]+]], r0,
  %add = add nsw i32 %rem, %div
; EABI: add r0, r1, [[div]]
; DARWIN: add r0{{.*}}[[sum]]
; WINDOWS: add [[rem]], [[div]]
  ret i32 %add
}

; RUN: llc < %s -O0 -relocation-model=pic -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -ppc-late-peephole=true | FileCheck %s --check-prefix=ELF64

define i32 @t1(i8 signext %a) nounwind {
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i32 @t2(i8 zeroext %a) nounwind {
  %1 = zext i8 %a to i32
  ret i32 %1
}

define i32 @t3(i16 signext %a) nounwind {
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i32 @t4(i16 zeroext %a) nounwind {
  %1 = zext i16 %a to i32
  ret i32 %1
}

define void @foo(i8 %a, i16 %b) nounwind {
; ELF64: foo
  %1 = call i32 @t1(i8 signext %a)
; ELF64: extsb
  %2 = call i32 @t2(i8 zeroext %a)
; ELF64: clrldi {{[0-9]+}}, {{[0-9]+}}, 56
  %3 = call i32 @t3(i16 signext %b)
; ELF64: extsh
  %4 = call i32 @t4(i16 zeroext %b)
; ELF64: clrldi {{[0-9]+}}, {{[0-9]+}}, 48

;; A few test to check materialization
  %5 = call i32 @t2(i8 zeroext 255)
; ELF64: li 3, 255
; ELF64-NOT: clrldi
  %6 = call i32 @t4(i16 zeroext 65535)
; ELF64: lis 3, 0
; ELF64: ori 3, 3, 65535
; ELF64: clrldi 3, 3, 48
; ELF64: bl t4
  ret void
}

define void @foo2() nounwind {
  %1 = call signext i16 @t5()
  %2 = call zeroext i16 @t6()
  %3 = call signext i8 @t7()
  %4 = call zeroext i8 @t8()
  ret void
}

declare signext i16 @t5();
declare zeroext i16 @t6();
declare signext i8 @t7();
declare zeroext i8 @t8();

define i32 @t10(i32 %argc, ptr nocapture %argv) nounwind {
entry:
; ELF64: t10
  %call = call i32 @bar(i8 zeroext 0, i8 zeroext -8, i8 zeroext -69, i8 zeroext 28, i8 zeroext 40, i8 zeroext -70)
; ELF64: li 3, 0
; ELF64: li 4, 248
; ELF64: li 5, 187
; ELF64: li 6, 28
; ELF64: li 7, 40
; ELF64: li 8, 186
; ELF64-NOT: clrldi
; ELF64: bl bar
  ret i32 0
}

declare i32 @bar(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext)

define i32 @bar0(i32 %i) nounwind {
  ret i32 0
}

; Function pointers are not yet implemented.
;define void @foo3() uwtable {
;  %fptr = alloca ptr, align 8
;  store ptr @bar0, ptr %fptr, align 8
;  %1 = load ptr, ptr %fptr, align 8
;  %call = call i32 %1(i32 0)
;  ret void
;}

; Intrinsic calls not yet implemented, and udiv isn't one for PPC anyway.
;define i32 @LibCall(i32 %a, i32 %b) {
;entry:
;        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
;        ret i32 %tmp1
;}

declare void @float_foo(float %f)

define void @float_const() nounwind {
entry:
; ELF64: float_const
  call void @float_foo(float 0x401C666660000000)
; ELF64: addis [[REG:[0-9]+]], 2, .LCPI[[SUF:[0-9_]+]]@toc@ha
; ELF64: lfs 1, .LCPI[[SUF]]@toc@l([[REG]])
  ret void
}

define void @float_reg(float %dummy, float %f) nounwind {
entry:
; ELF64: float_reg
  call void @float_foo(float %f)
; ELF64: fmr 1, 2
  ret void
}

declare void @double_foo(double %d)

define void @double_const() nounwind {
entry:
; ELF64: double_const
  call void @double_foo(double 0x1397723CCABD0000401C666660000000)
; ELF64: addis [[REG2:[0-9]+]], 2, .LCPI[[SUF2:[0-9_]+]]@toc@ha
; ELF64: lfd 1, .LCPI[[SUF2]]@toc@l([[REG2]])
  ret void
}

define void @double_reg(double %dummy, double %d) nounwind {
entry:
; ELF64: double_reg
  call void @double_foo(double %d)
; ELF64: fmr 1, 2
  ret void
}

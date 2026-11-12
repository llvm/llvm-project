; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM --check-prefix=ARM-MACHO
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM --check-prefix=ARM-ELF
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB

; Very basic fast-isel functionality.
define i32 @test0(i32 %a, i32 %b) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr
  store i32 %b, ptr %b.addr
  %tmp = load i32, ptr %a.addr
  %tmp1 = load i32, ptr %b.addr
  %add = add nsw i32 %tmp, %tmp1
  ret i32 %add
}

; Check truncate to bool
define void @test1(i32 %tmp) nounwind {
entry:
%tobool = trunc i32 %tmp to i1
br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
call void @test1(i32 0)
br label %if.end

if.end:                                           ; preds = %if.then, %entry
ret void
; ARM-LABEL: test1:
; ARM: tst r0, #1
; THUMB-LABEL: test1:
; THUMB: tst.w r0, #1
}

; Check some simple operations with immediates
define void @test2(i32 %tmp, ptr %ptr) nounwind {
; THUMB-LABEL: test2:
; ARM-LABEL: test2:

b1:
  %a = add i32 %tmp, 4096
  store i32 %a, ptr %ptr
  br label %b2

; THUMB: add.w {{.*}} #4096
; ARM: add {{.*}} #4096

b2:
  %b = add i32 %tmp, 4095
  store i32 %b, ptr %ptr
  br label %b3
; THUMB: addw {{.*}} #4095
; ARM: movw {{.*}} #4095
; ARM: add

b3:
  %c = or i32 %tmp, 4
  store i32 %c, ptr %ptr
  ret void

; THUMB: orr {{.*}} #4
; ARM: orr {{.*}} #4
}

define void @test3(i32 %tmp, ptr %ptr1, ptr %ptr2, ptr %ptr3) nounwind {
; THUMB-LABEL: test3:
; ARM-LABEL: test3:

bb1:
  %a1 = trunc i32 %tmp to i16
  %a2 = trunc i16 %a1 to i8
  %a3 = trunc i8 %a2 to i1
  %a4 = zext i1 %a3 to i8
  store i8 %a4, ptr %ptr3
  %a5 = zext i8 %a4 to i16
  store i16 %a5, ptr %ptr2
  %a6 = zext i16 %a5 to i32
  store i32 %a6, ptr %ptr1
  br label %bb2

; THUMB: and
; THUMB: strb
; THUMB: and{{.*}}, #255
; THUMB: strh
; THUMB: uxth
; ARM: and
; ARM: strb
; ARM: and{{.*}}, #255
; ARM: strh
; ARM: uxth

bb2:
  %b1 = trunc i32 %tmp to i16
  %b2 = trunc i16 %b1 to i8
  store i8 %b2, ptr %ptr3
  %b3 = sext i8 %b2 to i16
  store i16 %b3, ptr %ptr2
  %b4 = sext i16 %b3 to i32
  store i32 %b4, ptr %ptr1
  br label %bb3

; THUMB: strb
; THUMB: sxtb
; THUMB: strh
; THUMB: sxth
; ARM: strb
; ARM: sxtb
; ARM: strh
; ARM: sxth

bb3:
  %c1 = load i8, ptr %ptr3
  %c2 = load i16, ptr %ptr2
  %c3 = load i32, ptr %ptr1
  %c4 = zext i8 %c1 to i32
  %c5 = sext i16 %c2 to i32
  %c6 = add i32 %c4, %c5
  %c7 = sub i32 %c3, %c6
  store i32 %c7, ptr %ptr1
  ret void

; THUMB: ldrb
; THUMB: ldrh
; THUMB: and{{.*}}, #255
; THUMB: sxth
; THUMB: add
; THUMB: sub
; ARM: ldrb
; ARM: ldrh
; ARM: and{{.*}}, #255
; ARM: sxth
; ARM: add
; ARM: sub
}

; Check loads/stores with globals
@test4g = external global i32

define void @test4() {
  %a = load i32, ptr @test4g
  %b = add i32 %a, 1
  store i32 %b, ptr @test4g
  ret void


; Note that relocations are either movw/movt or constant pool
; loads. Different platforms will select different approaches.

; THUMB: {{(movw r0, :lower16:L_test4g\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_test4g\$non_lazy_ptr)?}}
; THUMB: ldr [[REG:r[0-9]+]], [r0]
; THUMB: ldr [[REG1:r[0-9]+]], [[[REG]]]
; THUMB: adds [[REG1]], #1
; THUMB: {{(movw r1, :lower16:L_test4g\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r1, :upper16:L_test4g\$non_lazy_ptr)?}}
; THUMB: ldr [[REG2:r[0-9]+]], [r1]
; THUMB: str [[REG1]], [[[REG2]]]

; ARM-MACHO: {{(movw r0, :lower16:L_test4g\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_test4g\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG:r[0-9]+]], :lower16:test4g
; ARM-ELF: movt [[REG]], :upper16:test4g

; ARM: ldr [[REG1:r[0-9]+]], [[[REG]]]
; ARM: add [[REG2:r[0-9]+]], [[REG1]], #1

; ARM-MACHO: {{(movw r1, :lower16:L_test4g\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r1, :upper16:L_test4g\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG3:r[0-9]+]], [r1]

; ARM-ELF: movw [[REG3:r[0-9]+]], :lower16:test4g
; ARM-ELF: movt [[REG3]], :upper16:test4g

; ARM: str [[REG2]], [[[REG3]]]
}

; ARM: @urem_fold
; THUMB: @urem_fold
; ARM: and r0, r0, #31
; THUMB: and r0, r0, #31
define i32 @urem_fold(i32 %a) nounwind {
  %rem = urem i32 %a, 32
  ret i32 %rem
}

define i32 @trap_intrinsic() noreturn nounwind  {
entry:
; ARM: @trap_intrinsic
; THUMB: @trap_intrinsic
; ARM: trap
; THUMB: trap
  tail call void @llvm.trap( )
  unreachable
}

declare void @llvm.trap() nounwind

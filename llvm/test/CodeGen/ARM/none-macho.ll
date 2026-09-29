; RUN: llc -mtriple=thumbv7m-none-macho %s -o - -relocation-model=pic -frame-pointer=all | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NON-FAST
; RUN: llc -mtriple=thumbv7m-none-macho -O0 %s -o - -relocation-model=pic -frame-pointer=all | FileCheck %s
; RUN: llc -mtriple=thumbv7m-none-macho -filetype=obj %s -o /dev/null

@var = external global i32

define i32 @test_litpool() minsize {
; CHECK-LABEL: test_litpool:
  %val = load i32, ptr @var
  ret i32 %val

  ; Lit-pool entries need to produce a "$non_lazy_ptr" version of the symbol.
; CHECK: LCPI0_0:
; CHECK-NEXT: .long L_var$non_lazy_ptr-(LPC0_0+4)
}

define i32 @test_movw_movt() {
; CHECK-LABEL: test_movw_movt:
  %val = load i32, ptr @var
  ret i32 %val

  ; movw/movt should also address their symbols MachO-style
; CHECK: movw [[RTMP:r[0-9]+]], :lower16:(L_var$non_lazy_ptr-(LPC1_0+4))
; CHECK: movt [[RTMP]], :upper16:(L_var$non_lazy_ptr-(LPC1_0+4))
; CHECK: LPC1_0:
; CHECK: add [[RTMP]], pc
}

declare void @llvm.trap()

define void @test_trap() {
; CHECK-LABEL: test_trap:

  ; Bare-metal MachO gets compiled on top of normal MachO toolchain which
  ; understands trap natively.
  call void @llvm.trap()
; CHECK: trap

  ret void
}

define i32 @test_frame_ptr() {
; CHECK-LABEL: test_frame_ptr:
  call void @test_trap()

  ; Frame pointer is r7.
; CHECK: mov r7, sp
  ret i32 42
}

%big_arr = type [8 x i32]
define void @test_two_areas(ptr %addr) {
; CHECK-LABEL: test_two_areas:
  %val = load %big_arr, ptr %addr
  call void @test_trap()
  store %big_arr %val, ptr %addr

  ; This goes with the choice of r7 as FP (largely). FP and LR have to be stored
  ; consecutively on the stack for the frame record to be valid, which means we
  ; need the 2 register-save areas employed by iOS.
; CHECK-NON-FAST: push {r4, r5, r6, r7, lr}
; CHECK-NON-FAST: push.w {r8, r9, r10, r11}
; ...
; CHECK-NON-FAST: pop.w {r8, r9, r10, r11}
; CHECK-NON-FAST: pop {r4, r5, r6, r7, pc}
  ret void
}

define void @test_tail_call() {
; CHECK-LABEL: test_tail_call:
  tail call void @test_trap()

  ; Tail calls should be available and use Thumb2 branch.
; CHECK: b.w _test_trap
  ret void
}

define float @test_softfloat_calls(float %in) {
; CHECK-LABEL: test_softfloat_calls:
  %sum = fadd float %in, %in

  ; Soft-float calls should be GNU-style rather than RTABI and should not be the
  ; *vfp variants used for ARMv6 iOS.
; CHECK: bl ___addsf3{{$}}
  ret float %sum
}

  ; Even bare-metal PIC needs GOT-like behaviour, in principle. Depends a bit on
  ; the use-case of course, but LLVM doesn't know what that is.
; CHECK: non_lazy_symbol_pointers
; CHECK: L_var$non_lazy_ptr:
; CHECK-NEXT:   .indirect_symbol _var

  ; All MachO objects should have this to give the linker leeway in removing
  ; dead code.
; CHECK: .subsections_via_symbols

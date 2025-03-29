; RUN: llc < %s -mtriple=avr -filetype=asm | FileCheck %s -check-prefix=CHECK-ASM
; RUN: llc < %s -mtriple=avr -filetype=obj | llvm-objdump -Dr - \
; RUN:                                   | FileCheck %s -check-prefix=CHECK-OBJ

; Somewhat pruned test case from rustc using trait objects

%TraitObjectA = type {}

; CHECK-ASM-LABEL: vtable.0:
; CHECK-ASM-NEXT: .short pm(drop_in_place2)
; CHECK-ASM-NEXT: .short 0
; CHECK-ASM-NEXT: .short 1
; CHECK-ASM-NEXT: .short pm(trait_fn2)

; CHECK-OBJ-LABEL: <vtable.0>:
; CHECK-OBJ-NEXT: 00 00
; CHECK-OBJ-NEXT: R_AVR_16_PM  .text
; CHECK-OBJ-NEXT: 00 00
; CHECK-OBJ-NEXT: 01 00 00 00
; CHECK-OBJ-NEXT: R_AVR_16_PM  .text
@vtable.0 = private constant {
    ptr addrspace(1),
    i16,
    i16,
    ptr addrspace(1)
  } {
    ptr addrspace(1)
      @drop_in_place2,
    i16 0,
    i16 1,
    ptr addrspace(1)
      @trait_fn2
  }, align 1

; CHECK-ASM-LABEL: vtable.1:
; CHECK-ASM-NEXT: .short pm(drop_in_place1)
; CHECK-ASM-NEXT: .short 0
; CHECK-ASM-NEXT: .short 1
; CHECK-ASM-NEXT: .short pm(trait_fn1)

; CHECK-OBJ-LABEL: <vtable.1>:
; CHECK-OBJ-NEXT: 00 00
; CHECK-OBJ-NEXT: R_AVR_16_PM  .text
; CHECK-OBJ-NEXT: 00 00
; CHECK-OBJ-NEXT: 01 00 00 00
; CHECK-OBJ-NEXT: R_AVR_16_PM  .text
@vtable.1 = private constant {
    ptr addrspace(1),
    i16,
    i16,
    ptr addrspace(1)
  } {
    ptr addrspace(1)
      @drop_in_place1,
    i16 0,
    i16 1,
    ptr addrspace(1)
      @trait_fn1
  }, align 1

define internal fastcc i8 @TraitObjectA_method(i1 zeroext %choice) addrspace(1) {
start:
  %b = alloca %TraitObjectA, align 1

  %c = select i1 %choice, ptr @vtable.0,
    ptr @vtable.1
  %b2 = bitcast ptr %b to ptr

  %res = call fastcc addrspace(1) i8 @call_trait_object(ptr nonnull align 1 %b2, ptr noalias readonly align 1 dereferenceable(6) %c)
  ret i8 %res
}

define internal fastcc i8 @call_trait_object(ptr nonnull align 1 %a, ptr noalias nocapture readonly align 1 dereferenceable(6) %b) addrspace(1) {
start:
  %b2 = getelementptr inbounds [3 x i16], ptr %b, i16 0, i16 3
  %c = bitcast ptr %b2 to ptr
  %d = load ptr addrspace(1), ptr %c, align 1, !invariant.load !1, !nonnull !1
  %res = tail call addrspace(1) i8 %d(ptr nonnull align 1 %a)
  ret i8 %res
}

define internal void @drop_in_place1(ptr nocapture %a) addrspace(1) {
start:
  ret void
}

define internal i8 @trait_fn1(ptr noalias nocapture nonnull readonly align 1 %self) addrspace(1) {
start:
  ret i8 89
}

define internal void @drop_in_place2(ptr nocapture %a) addrspace(1) {
start:
  ret void
}

define internal i8 @trait_fn2(ptr noalias nocapture nonnull readonly align 1 %self) addrspace(1) {
start:
  ret i8 79
}

!1 = !{}

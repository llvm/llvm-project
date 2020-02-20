; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu \
; RUN:    -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC

define i32 @fn() nounwind noinline uwtable "function-instrument"="xray-always" {
    %eventptr = alloca i8
    %eventsize = alloca i32
    %eventtype = alloca i16
    store i16 6, i16* %eventtype
    %type = load i16, i16* %eventtype
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.typedevent(i16 %type, i8* %eventptr, i32 %val)
    ; CHECK-LABEL: Lxray_typed_event_sled_0:
    ; CHECK:       .byte 0xeb, 0x14
    ; CHECK-NEXT:  pushq %rdi
    ; CHECK-NEXT:  pushq %rsi
    ; CHECK-NEXT:  pushq %rdx
    ; CHECK-NEXT:  movq %rdx, %rdi
    ; CHECK-NEXT:  movq %rcx, %rsi
    ; CHECK-NEXT:  movq %rax, %rdx
    ; CHECK-NEXT:  callq __xray_TypedEvent
    ; CHECK-NEXT:  popq %rdx
    ; CHECK-NEXT:  popq %rsi
    ; CHECK-NEXT:  popq %rdi

    ; PIC-LABEL: Lxray_typed_event_sled_0:
    ; PIC:       .byte 0xeb, 0x14
    ; PIC-NEXT:  pushq %rdi
    ; PIC-NEXT:  pushq %rsi
    ; PIC-NEXT:  pushq %rdx
    ; PIC-NEXT:  movq %rdx, %rdi
    ; PIC-NEXT:  movq %rcx, %rsi
    ; PIC-NEXT:  movq %rax, %rdx
    ; PIC-NEXT:  callq __xray_TypedEvent@PLT
    ; PIC-NEXT:  popq %rdx
    ; PIC-NEXT:  popq %rsi
    ; PIC-NEXT:  popq %rdi
    ret i32 0
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_typed_event_sled_0

declare void @llvm.xray.typedevent(i16, i8*, i32)

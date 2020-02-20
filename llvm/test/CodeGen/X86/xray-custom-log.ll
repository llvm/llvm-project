; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - \
; RUN:     -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC

define i32 @fn() nounwind noinline uwtable "function-instrument"="xray-always" {
    %eventptr = alloca i8
    %eventsize = alloca i32
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.customevent(i8* %eventptr, i32 %val)
    ; CHECK-LABEL: Lxray_event_sled_0:
    ; CHECK:       .byte 0xeb, 0x0f
    ; CHECK-NEXT:  pushq %rdi
    ; CHECK-NEXT:  pushq %rsi
    ; CHECK-NEXT:  movq %rcx, %rdi
    ; CHECK-NEXT:  movq %rax, %rsi
    ; CHECK-NEXT:  callq __xray_CustomEvent
    ; CHECK-NEXT:  popq %rsi
    ; CHECK-NEXT:  popq %rdi

    ; PIC-LABEL: Lxray_event_sled_0:
    ; PIC:       .byte 0xeb, 0x0f
    ; PIC-NEXT:  pushq %rdi
    ; PIC-NEXT:  pushq %rsi
    ; PIC-NEXT:  movq %rcx, %rdi
    ; PIC-NEXT:  movq %rax, %rsi
    ; PIC-NEXT:  callq __xray_CustomEvent@PLT
    ; PIC-NEXT:  popq %rsi
    ; PIC-NEXT:  popq %rdi
    ret i32 0
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_event_sled_0

declare void @llvm.xray.customevent(i8*, i32)

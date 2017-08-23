; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i32 @fn() nounwind noinline uwtable "function-instrument"="xray-always" {
    %eventptr = alloca i8
    %eventsize = alloca i32
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.customevent(i8* %eventptr, i32 %val)
    ; CHECK-LABEL: Lxray_event_sled_0:
    ; CHECK:       .byte 0xeb, 0x0f
    ; CHECK-NEXT:  pushq %rdi
    ; CHECK-NEXT:  movq {{.*}}, %rdi
    ; CHECK-NEXT:  pushq %rsi
    ; CHECK-NEXT:  movq {{.*}}, %rsi
    ; CHECK-NEXT:  callq __xray_CustomEvent
    ; CHECK-NEXT:  popq %rsi
    ; CHECK-NEXT:  popq %rdi
    ret i32 0
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_event_sled_0

declare void @llvm.xray.customevent(i8*, i32)

; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i32 @fn() nounwind noinline uwtable "function-instrument"="xray-always" {
    %eventptr = alloca i8
    %eventsize = alloca i32
    store i32 3, i32* %eventsize
    %val = load i32, i32* %eventsize
    call void @llvm.xray.customevent(i8* %eventptr, i32 %val)
    ; CHECK-LABEL: Lxray_event_sled_0:
    ; CHECK-NEXT:  .ascii "\353\024
    ; CHECK-NEXT:  pushq %rax
    ; CHECK-NEXT:  movq {{.*}}, %rdi
    ; CHECK-NEXT:  movq {{.*}}, %rsi
    ; CHECK-NEXT:  movabsq $__xray_CustomEvent, %rax
    ; CHECK-NEXT:  callq *%rax
    ; CHECK-NEXT:  popq %rax
    ret i32 0
}
; CHECK:       .section {{.*}}xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_event_sled_0

declare void @llvm.xray.customevent(i8*, i32)

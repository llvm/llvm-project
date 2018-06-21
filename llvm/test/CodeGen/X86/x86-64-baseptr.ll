; RUN: llc -mtriple=x86_64-pc-linux -stackrealign -stack-alignment=32 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 -stackrealign -stack-alignment=32 < %s | FileCheck -check-prefix=X32ABI %s
; This should run with NaCl as well ( -mtriple=x86_64-pc-nacl ) but currently doesn't due to PR22655

; Make sure the correct register gets set up as the base pointer
; This should be rbx for x64 and 64-bit NaCl and ebx for x32
; CHECK-LABEL: base
; CHECK: subq $32, %rsp
; CHECK: movq %rsp, %rbx
; X32ABI-LABEL: base
; X32ABI: subl $32, %esp
; X32ABI: movl %esp, %ebx
; NACL-LABEL: base
; NACL: subq $32, %rsp
; NACL: movq %rsp, %rbx

declare i32 @helper() nounwind
define void @base() #0 {
entry:
  %k = call i32 @helper()
  %a = alloca i32, i32 %k, align 4
  store i32 0, i32* %a, align 4
  ret void
}

attributes #0 = { nounwind uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"}

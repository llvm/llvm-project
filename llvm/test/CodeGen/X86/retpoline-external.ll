; RUN: llc -mtriple=x86_64-unknown < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X64
; RUN: llc -mtriple=x86_64-unknown -O0 < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X64FAST

; RUN: llc -mtriple=i686-unknown < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X86
; RUN: llc -mtriple=i686-unknown -O0 < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X86FAST

declare void @bar(i32)

; Test a simple indirect call and tail call.
define void @icall_reg(void (i32)* %fp, i32 %x) #0 {
entry:
  tail call void @bar(i32 %x)
  tail call void %fp(i32 %x)
  tail call void @bar(i32 %x)
  tail call void %fp(i32 %x)
  ret void
}

; X64-LABEL: icall_reg:
; X64-DAG:   movq %rdi, %[[fp:[^ ]*]]
; X64-DAG:   movl %esi, %[[x:[^ ]*]]
; X64:       movl %[[x]], %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       callq __llvm_external_retpoline_r11
; X64:       movl %[[x]], %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X64FAST-LABEL: icall_reg:
; X64FAST:       callq bar
; X64FAST:       callq __llvm_external_retpoline_r11
; X64FAST:       callq bar
; X64FAST:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X86-LABEL: icall_reg:
; X86-DAG:   movl 12(%esp), %[[fp:[^ ]*]]
; X86-DAG:   movl 16(%esp), %[[x:[^ ]*]]
; X86:       pushl %[[x]]
; X86:       calll bar
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[x]]
; X86:       calll __llvm_external_retpoline_eax
; X86:       pushl %[[x]]
; X86:       calll bar
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[x]]
; X86:       calll __llvm_external_retpoline_eax
; X86-NOT:   # TAILCALL

; X86FAST-LABEL: icall_reg:
; X86FAST:       calll bar
; X86FAST:       calll __llvm_external_retpoline_eax
; X86FAST:       calll bar
; X86FAST:       calll __llvm_external_retpoline_eax


@global_fp = external global void (i32)*

; Test an indirect call through a global variable.
define void @icall_global_fp(i32 %x, void (i32)** %fpp) #0 {
  %fp1 = load void (i32)*, void (i32)** @global_fp
  call void %fp1(i32 %x)
  %fp2 = load void (i32)*, void (i32)** @global_fp
  tail call void %fp2(i32 %x)
  ret void
}

; X64-LABEL: icall_global_fp:
; X64-DAG:   movl %edi, %[[x:[^ ]*]]
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       callq __llvm_external_retpoline_r11
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X64FAST-LABEL: icall_global_fp:
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       callq __llvm_external_retpoline_r11
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X86-LABEL: icall_global_fp:
; X86:       movl global_fp, %eax
; X86:       pushl 4(%esp)
; X86:       calll __llvm_external_retpoline_eax
; X86:       addl $4, %esp
; X86:       movl global_fp, %eax
; X86:       jmp __llvm_external_retpoline_eax # TAILCALL

; X86FAST-LABEL: icall_global_fp:
; X86FAST:       calll __llvm_external_retpoline_eax
; X86FAST:       jmp __llvm_external_retpoline_eax # TAILCALL


%struct.Foo = type { void (%struct.Foo*)** }

; Test an indirect call through a vtable.
define void @vcall(%struct.Foo* %obj) #0 {
  %vptr_field = getelementptr %struct.Foo, %struct.Foo* %obj, i32 0, i32 0
  %vptr = load void (%struct.Foo*)**, void (%struct.Foo*)*** %vptr_field
  %vslot = getelementptr void(%struct.Foo*)*, void(%struct.Foo*)** %vptr, i32 1
  %fp = load void(%struct.Foo*)*, void(%struct.Foo*)** %vslot
  tail call void %fp(%struct.Foo* %obj)
  tail call void %fp(%struct.Foo* %obj)
  ret void
}

; X64-LABEL: vcall:
; X64:       movq %rdi, %[[obj:[^ ]*]]
; X64:       movq (%[[obj]]), %[[vptr:[^ ]*]]
; X64:       movq 8(%[[vptr]]), %[[fp:[^ ]*]]
; X64:       movq %[[fp]], %r11
; X64:       callq __llvm_external_retpoline_r11
; X64-DAG:   movq %[[obj]], %rdi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X64FAST-LABEL: vcall:
; X64FAST:       callq __llvm_external_retpoline_r11
; X64FAST:       jmp __llvm_external_retpoline_r11 # TAILCALL

; X86-LABEL: vcall:
; X86:       movl 8(%esp), %[[obj:[^ ]*]]
; X86:       movl (%[[obj]]), %[[vptr:[^ ]*]]
; X86:       movl 4(%[[vptr]]), %[[fp:[^ ]*]]
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[obj]]
; X86:       calll __llvm_external_retpoline_eax
; X86:       addl $4, %esp
; X86:       movl %[[fp]], %eax
; X86:       jmp __llvm_external_retpoline_eax # TAILCALL

; X86FAST-LABEL: vcall:
; X86FAST:       calll __llvm_external_retpoline_eax
; X86FAST:       jmp __llvm_external_retpoline_eax # TAILCALL


declare void @direct_callee()

define void @direct_tail() #0 {
  tail call void @direct_callee()
  ret void
}

; X64-LABEL: direct_tail:
; X64:       jmp direct_callee # TAILCALL
; X64FAST-LABEL: direct_tail:
; X64FAST:   jmp direct_callee # TAILCALL
; X86-LABEL: direct_tail:
; X86:       jmp direct_callee # TAILCALL
; X86FAST-LABEL: direct_tail:
; X86FAST:   jmp direct_callee # TAILCALL


; Lastly check that no thunks were emitted.
; X64-NOT: __{{.*}}_retpoline_{{.*}}:
; X64FAST-NOT: __{{.*}}_retpoline_{{.*}}:
; X86-NOT: __{{.*}}_retpoline_{{.*}}:
; X86FAST-NOT: __{{.*}}_retpoline_{{.*}}:


attributes #0 = { "target-features"="+retpoline-external-thunk" }

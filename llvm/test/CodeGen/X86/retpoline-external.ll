; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X64
; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown -O0 < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X64FAST

; RUN: llc -verify-machineinstrs -mtriple=i686-unknown < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X86
; RUN: llc -verify-machineinstrs -mtriple=i686-unknown -O0 < %s | FileCheck %s --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" --check-prefix=X86FAST

declare dso_local void @bar(i32)

; Test a simple indirect call and tail call.
define void @icall_reg(ptr %fp, i32 %x) #0 {
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
; X64:       movl %esi, %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       callq __x86_indirect_thunk_r11
; X64:       movl %[[x]], %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X64FAST-LABEL: icall_reg:
; X64FAST:       callq bar
; X64FAST:       callq __x86_indirect_thunk_r11
; X64FAST:       callq bar
; X64FAST:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X86-LABEL: icall_reg:
; X86-DAG:   movl 12(%esp), %[[fp:[^ ]*]]
; X86-DAG:   movl 16(%esp), %[[x:[^ ]*]]
; X86:       pushl %[[x]]
; X86:       calll bar
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[x]]
; X86:       calll __x86_indirect_thunk_eax
; X86:       pushl %[[x]]
; X86:       calll bar
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[x]]
; X86:       calll __x86_indirect_thunk_eax
; X86-NOT:   # TAILCALL

; X86FAST-LABEL: icall_reg:
; X86FAST:       calll bar
; X86FAST:       calll __x86_indirect_thunk_eax
; X86FAST:       calll bar
; X86FAST:       calll __x86_indirect_thunk_eax


@global_fp = external dso_local global ptr

; Test an indirect call through a global variable.
define void @icall_global_fp(i32 %x, ptr %fpp) #0 {
  %fp1 = load ptr, ptr @global_fp
  call void %fp1(i32 %x)
  %fp2 = load ptr, ptr @global_fp
  tail call void %fp2(i32 %x)
  ret void
}

; X64-LABEL: icall_global_fp:
; X64-DAG:   movl %edi, %[[x:[^ ]*]]
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       callq __x86_indirect_thunk_r11
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X64FAST-LABEL: icall_global_fp:
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       callq __x86_indirect_thunk_r11
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X86-LABEL: icall_global_fp:
; X86:       movl global_fp, %eax
; X86:       pushl 4(%esp)
; X86:       calll __x86_indirect_thunk_eax
; X86:       addl $4, %esp
; X86:       movl global_fp, %eax
; X86:       jmp __x86_indirect_thunk_eax # TAILCALL

; X86FAST-LABEL: icall_global_fp:
; X86FAST:       calll __x86_indirect_thunk_eax
; X86FAST:       jmp __x86_indirect_thunk_eax # TAILCALL


%struct.Foo = type { ptr }

; Test an indirect call through a vtable.
define void @vcall(ptr %obj) #0 {
  %vptr = load ptr, ptr %obj
  %vslot = getelementptr ptr, ptr %vptr, i32 1
  %fp = load ptr, ptr %vslot
  tail call void %fp(ptr %obj)
  tail call void %fp(ptr %obj)
  ret void
}

; X64-LABEL: vcall:
; X64:       movq %rdi, %[[obj:[^ ]*]]
; X64:       movq (%rdi), %[[vptr:[^ ]*]]
; X64:       movq 8(%[[vptr]]), %[[fp:[^ ]*]]
; X64:       movq %[[fp]], %r11
; X64:       callq __x86_indirect_thunk_r11
; X64-DAG:   movq %[[obj]], %rdi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X64FAST-LABEL: vcall:
; X64FAST:       callq __x86_indirect_thunk_r11
; X64FAST:       jmp __x86_indirect_thunk_r11 # TAILCALL

; X86-LABEL: vcall:
; X86:       movl 8(%esp), %[[obj:[^ ]*]]
; X86:       movl (%[[obj]]), %[[vptr:[^ ]*]]
; X86:       movl 4(%[[vptr]]), %[[fp:[^ ]*]]
; X86:       movl %[[fp]], %eax
; X86:       pushl %[[obj]]
; X86:       calll __x86_indirect_thunk_eax
; X86:       addl $4, %esp
; X86:       movl %[[fp]], %eax
; X86:       jmp __x86_indirect_thunk_eax # TAILCALL

; X86FAST-LABEL: vcall:
; X86FAST:       calll __x86_indirect_thunk_eax
; X86FAST:       jmp __x86_indirect_thunk_eax # TAILCALL


declare dso_local void @direct_callee()

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


attributes #0 = { "target-features"="+retpoline-indirect-calls,+retpoline-external-thunk" }

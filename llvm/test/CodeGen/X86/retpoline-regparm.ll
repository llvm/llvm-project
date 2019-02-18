; RUN: llc -verify-machineinstrs -mtriple=i686-linux < %s | FileCheck --implicit-check-not="jmp.*\*" --implicit-check-not="call.*\*" %s

; Test 32-bit retpoline when -mregparm=3 is used. This case is interesting
; because there are no available scratch registers.  The Linux kernel builds
; with -mregparm=3, so we need to support it.  TCO should fail because we need
; to restore EDI.

define void @call_edi(void (i32, i32, i32)* %fp) #0 {
entry:
  tail call void %fp(i32 inreg 0, i32 inreg 0, i32 inreg 0)
  ret void
}

; CHECK-LABEL: call_edi:
;     EDI is used, so it must be saved.
; CHECK: pushl %edi
; CHECK-DAG: xorl %eax, %eax
; CHECK-DAG: xorl %edx, %edx
; CHECK-DAG: xorl %ecx, %ecx
; CHECK-DAG: movl {{.*}}, %edi
; CHECK: calll __llvm_retpoline_edi
; CHECK: popl %edi
; CHECK: retl

define void @edi_external(void (i32, i32, i32)* %fp) #1 {
entry:
  tail call void %fp(i32 inreg 0, i32 inreg 0, i32 inreg 0)
  ret void
}

; CHECK-LABEL: edi_external:
; CHECK: pushl %edi
; CHECK-DAG: xorl %eax, %eax
; CHECK-DAG: xorl %edx, %edx
; CHECK-DAG: xorl %ecx, %ecx
; CHECK-DAG: movl {{.*}}, %edi
; CHECK: calll __x86_indirect_thunk_edi
; CHECK: popl %edi
; CHECK: retl

attributes #0 = { "target-features"="+retpoline-indirect-calls" }
attributes #1 = { "target-features"="+retpoline-indirect-calls,+retpoline-external-thunk" }

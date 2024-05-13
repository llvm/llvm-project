; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mattr=+secure-plt -relocation-model=pic | FileCheck -check-prefix=SECURE-PLT-TLS %s

@a = thread_local local_unnamed_addr global i32 6, align 4
define i32 @main() local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr @a, align 4
  ret i32 %0
}


!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}

; SECURE-PLT-TLS:       mflr 30
; SECURE-PLT-TLS-NEXT:  addis 30, 30, .LTOC-.L0$pb@ha
; SECURE-PLT-TLS-NEXT:  addi 30, 30, .LTOC-.L0$pb@l
; SECURE-PLT-TLS-NEXT:  bl .L{{.*}}
; SECURE-PLT-TLS:       bl __tls_get_addr(a@tlsgd)@PLT+32768
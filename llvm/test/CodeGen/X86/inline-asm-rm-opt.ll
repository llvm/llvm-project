; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: test_1:
; CHECK:         #APP
; CHECK-NEXT:    pushq %rdi
; CHECK-NEXT:    popfq
; CHECK-NEXT:    #NO_APP
define dso_local void @test_1(i64 noundef %flags) local_unnamed_addr {
entry:
  tail call void asm sideeffect "push $0 ; popf", "rm,~{dirflag},~{fpsr},~{flags}"(i64 %flags)
  ret void
}

; CHECK-LABEL: test_2:
; CHECK:         #APP
; CHECK-NEXT:    pushfq
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    #NO_APP
define dso_local i64 @test_2() local_unnamed_addr {
entry:
  %0 = tail call i64 asm sideeffect "pushf ; pop $0", "=rm,~{dirflag},~{fpsr},~{flags}"()
  ret i64 %0
}

; CHECK-LABEL: test_3:
; CHECK:         #APP
; CHECK-NEXT:    pushq %rdi
; CHECK-NEXT:    popfq
; CHECK-NEXT:    #NO_APP
define dso_local void @test_3(i64 noundef %flags) local_unnamed_addr {
entry:
  tail call void asm sideeffect "push $0 ; popf", "imr,~{dirflag},~{fpsr},~{flags}"(i64 %flags)
  ret void
}

; CHECK-LABEL: test_4:
; CHECK:         #APP
; CHECK-NEXT:    pushfq
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    #NO_APP
define dso_local i64 @test_4() local_unnamed_addr {
entry:
  %0 = tail call i64 asm sideeffect "pushf ; pop $0", "=imr,~{dirflag},~{fpsr},~{flags}"()
  ret i64 %0
}

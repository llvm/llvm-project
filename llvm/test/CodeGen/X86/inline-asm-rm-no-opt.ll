; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 < %s | FileCheck %s

; CHECK-LABEL: test_1:
; CHECK:         #APP
; CHECK-NEXT:    pushq -16(%{{.*}})
; CHECK-NEXT:    popfq
; CHECK-NEXT:    #NO_APP
define dso_local void @test_1(i64 noundef %flags) {
entry:
  %flags.addr = alloca i64, align 8
  store i64 %flags, ptr %flags.addr, align 8
  %0 = load i64, ptr %flags.addr, align 8
  call void asm sideeffect "push $0 ; popf", "rm,~{dirflag},~{fpsr},~{flags}"(i64 %0)
  ret void
}

; CHECK-LABEL: test_2:
; CHECK:         #APP
; CHECK-NEXT:    pushfq
; CHECK-NEXT:    popq -8(%{{.*}})
; CHECK-NEXT:    #NO_APP
define dso_local i64 @test_2() {
entry:
  %out = alloca i64, align 8
  call void asm sideeffect "pushf ; pop $0", "=*rm,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %out)
  %0 = load i64, ptr %out, align 8
  ret i64 %0
}

; CHECK-LABEL: test_3:
; CHECK:         #APP
; CHECK-NEXT:    pushq -16(%{{.*}})
; CHECK-NEXT:    popfq
; CHECK-NEXT:    #NO_APP
define dso_local void @test_3(i64 noundef %flags) {
entry:
  %flags.addr = alloca i64, align 8
  store i64 %flags, ptr %flags.addr, align 8
  %0 = load i64, ptr %flags.addr, align 8
  call void asm sideeffect "push $0 ; popf", "imr,~{dirflag},~{fpsr},~{flags}"(i64 %0)
  ret void
}

; CHECK-LABEL: test_4:
; CHECK:         #APP
; CHECK-NEXT:    pushfq
; CHECK-NEXT:    popq -8(%{{.*}})
; CHECK-NEXT:    #NO_APP
define dso_local i64 @test_4() {
entry:
  %out = alloca i64, align 8
  call void asm sideeffect "pushf ; pop $0", "=*imr,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i64) %out)
  %0 = load i64, ptr %out, align 8
  ret i64 %0
}

;; Check that the generated memory references do not contain the `offset`
;; operator. Use `-no-integrated-as` to disable AsmParser formatting.
; RUN: llc -no-integrated-as -x86-asm-syntax=intel < %s | FileCheck %s

;; This file was compiled from clang/test/CodeGen/ms-inline-asm-functions.c,
;; using the following command line:
;;
;; bin/clang -cc1 -internal-isystem lib/clang/17/include -nostdsysteminc \
;;           ../llvm-project/clang/test/CodeGen/ms-inline-asm-functions.c \
;;           -triple i386-pc-windows-msvc -fms-extensions -S -o out.ll

source_filename = "/llvm-project/clang/test/CodeGen/ms-inline-asm-functions.c"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

@kptr = dso_local global ptr null, align 4

; Function Attrs: noinline nounwind optnone
define dso_local i32 @foo() #0 {
entry:
  %r = alloca ptr, align 4
  %call = call ptr @gptr()
  store ptr %call, ptr %r, align 4
  %0 = call i32 asm sideeffect inteldialect "call ${1:P}\0A\09call $2\0A\09call ${3:P}\0A\09call $4", "={eax},*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32 (i32)) @k, ptr elementtype(ptr) %r, ptr elementtype(i32 (i32)) @kimport, ptr elementtype(ptr) @kptr) #3, !srcloc !4
  ; CHECK-LABEL: _foo:
  ; CHECK:      #APP
  ; CHECK-NEXT: .intel_syntax
  ; CHECK-NEXT: call _k
  ; CHECK-NEXT: call [e{{([a-z]{2})}}]
  ; CHECK-NEXT: call [e{{([a-z]{2})}}]
  ; CHECK-NEXT: call [_kptr]
  ; CHECK-NEXT: .att_syntax
  ; CHECK-NEXT: #NO_APP
  ret i32 %0
}

declare dso_local ptr @gptr()

declare dso_local i32 @k(i32 noundef)

declare dllimport i32 @kimport(i32 noundef)

; Function Attrs: noinline nounwind optnone
define dso_local i32 @bar() #0 {
entry:
  %0 = call i32 asm sideeffect inteldialect "jmp ${1:P}\0A\09ja ${2:P}\0A\09JAE ${3:P}\0A\09LOOP ${4:P}\0A\09loope ${5:P}\0A\09loopne ${6:P}", "={eax},*m,*m,*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32 (i32)) @k, ptr elementtype(i32 (i32)) @k, ptr elementtype(i32 (i32)) @k, ptr elementtype(i32 (i32)) @k, ptr elementtype(i32 (i32)) @k, ptr elementtype(i32 (i32)) @k) #3, !srcloc !5
  ; CHECK-LABEL: _bar:
  ; CHECK:      #APP
  ; CHECK-NEXT: .intel_syntax
  ; CHECK-NEXT: jmp _k
  ; CHECK-NEXT: ja [_k]
  ; CHECK-NEXT: JAE [_k]
  ; CHECK-NEXT: LOOP [_k]
  ; CHECK-NEXT: loope [_k]
  ; CHECK-NEXT: loopne [_k]
  ; CHECK-NEXT: .att_syntax
  ; CHECK-NEXT: #NO_APP
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @baz() #0 {
entry:
  %0 = call i32 asm sideeffect inteldialect "mov eax, $1\0A\09mov eax, $2", "=&{eax},*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32 (i32)) @k, ptr elementtype(ptr) @kptr) #3, !srcloc !6
  ; CHECK-LABEL: _baz:
  ; CHECK:      #APP
  ; CHECK-NEXT: .intel_syntax
  ; CHECK-NEXT: mov eax, [_k]
  ; CHECK-NEXT: mov eax, [_kptr]
  ; CHECK-NEXT: .att_syntax
  ; CHECK-NEXT: #NO_APP
  ret i32 %0
}

; Function Attrs: naked noinline nounwind optnone
define dso_local void @naked() #2 {
entry:
  call void asm sideeffect inteldialect "pusha\0A\09call ${0:P}\0A\09popa\0A\09ret", "*m,~{eax},~{ebp},~{ebx},~{ecx},~{edi},~{edx},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32 (i32)) @k) #3, !srcloc !7
  ; CHECK-LABEL: _naked:
  ; CHECK:      #APP
  ; CHECK-NEXT: .intel_syntax
  ; CHECK-NEXT: pusha
  ; CHECK-NEXT: call _k
  ; CHECK-NEXT: popa
  ; CHECK-NEXT: ret
  ; CHECK-NEXT: .att_syntax
  ; CHECK-NEXT: #NO_APP
  unreachable
}

attributes #0 = { noinline nounwind optnone }
attributes #2 = { naked noinline nounwind optnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, !"MaxTLSAlign", i32 65536}
!3 = !{!"clang version 17.0.0"}
!4 = !{i64 528}
!5 = !{i64 892}
!6 = !{i64 1183}
!7 = !{i64 1451}

; RUN: llc < %s -mtriple=sparc   -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=medium | FileCheck --check-prefix=abs44 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=static -code-model=large  | FileCheck --check-prefix=abs64 %s
; RUN: llc < %s -mtriple=sparc   -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v8pic32 %s
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v9pic32 %s

define void @func1() #0 {
entry:
  ret void
}

define void @test() #0 {
entry:
  %pFunc = alloca void (...)*, align 4
  store void (...)* bitcast (void ()* @func1 to void (...)*), void (...)** %pFunc, align 4
  %0 = load void (...)*, void (...)** %pFunc, align 4
  %callee.knr.cast = bitcast void (...)* %0 to void ()*
  call void %callee.knr.cast()

; abs32-LABEL:   test
; abs32:          sethi %hi(func1), %i0
; abs32:          add %i0, %lo(func1), %i1
; abs32:          call %i0+%lo(func1)

; abs44-LABEL:   test
; abs44:          sethi %h44(func1), %i0
; abs44:          add %i0, %m44(func1), %i0
; abs44:          sllx %i0, 12, %i0
; abs44:          add %i0, %l44(func1), %i1
; abs44:          call %i0+%l44(func1)

; abs64-LABEL:   test
; abs64:          sethi %hi(func1), %i0
; abs64:          add %i0, %lo(func1), %i0
; abs64:          sethi %hh(func1), %i1
; abs64:          add %i1, %hm(func1), %i1

; v8pic32-LABEL: test
; v8pic32:        sethi %hi(func1), %i1
; v8pic32:        add %i1, %lo(func1), %i1
; v8pic32:        ld [%i0+%i1], %i0

; v9pic32-LABEL: test
; v9pic32:        sethi %hi(func1), %i1
; v9pic32:        add %i1, %lo(func1), %i1
; v9pic32:        ldx [%i0+%i1], %i0
; v9pic32:        call %i0

  ret void
}

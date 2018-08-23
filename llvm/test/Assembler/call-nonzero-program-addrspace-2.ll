; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llvm-as %s -data-layout=P42 -o - | llvm-dis - -o - | FileCheck %s -check-prefix PROGAS42

; Check that numbered variables in a nonzero program address space 200 can be used in a call instruction

define i8 @test_unnamed(i8(i32)*, i8(i32) addrspace(42)*) {
  ; Calls with explicit address spaces are fine:
  call addrspace(0) i8 %0(i32 0)
  call addrspace(42) i8 %1(i32 0)
  ; this call is fine if the program address space is 42
  call i8 %1(i32 0)
  ; CHECK: call-nonzero-program-addrspace-2.ll:[[@LINE-1]]:11: error: '%1' defined with type 'i8 (i32) addrspace(42)*' but expected 'i8 (i32)*'
  ret i8 0
}

; PROGAS42:       target datalayout = "P42"
; PROGAS42:       define i8 @test_unnamed(i8 (i32)*, i8 (i32) addrspace(42)*) addrspace(42) {
; PROGAS42-NEXT:    %3 = call addrspace(0) i8 %0(i32 0)
; PROGAS42-NEXT:    %4 = call addrspace(42) i8 %1(i32 0)
; PROGAS42-NEXT:    %5 = call addrspace(42) i8 %1(i32 0)
; PROGAS42-NEXT:    ret i8 0
; PROGAS42-NEXT:  }

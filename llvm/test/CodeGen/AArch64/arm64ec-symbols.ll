; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -t %t.o | FileCheck --check-prefix=SYM %s

declare void @func() nounwind;

define void @caller() nounwind {
  call void @func()
  ret void
}

; CHECK:      .weak_anti_dep  caller
; CHECK-NEXT: caller = "#caller"{{$}}

; CHECK:      .weak_anti_dep  func
; CHECK-NEXT: func = "#func"{{$}}
; CHECK-NEXT: .weak_anti_dep  "#func"
; CHECK-NEXT: "#func" = "#func$exit_thunk"{{$}}

; SYM:       [ 8](sec  4)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #caller
; SYM:       [21](sec  7)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #func$exit_thunk
; SYM:       [33](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 caller
; SYM-NEXT:  AUX indx 8 srch 4
; SYM-NEXT:  [35](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #func
; SYM-NEXT:  AUX indx 21 srch 4
; SYM:       [39](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 func
; SYM-NEXT:  AUX indx 35 srch 4

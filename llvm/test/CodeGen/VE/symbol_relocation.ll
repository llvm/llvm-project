; RUN: llc -filetype=obj -mtriple=ve -o - %s |  llvm-objdump - -d -r \
; RUN:     | FileCheck %s
; RUN: llc -filetype=obj -mtriple=ve -relocation-model=pic -o - %s \
; RUN:     |  llvm-objdump - -d -r | FileCheck %s -check-prefix=PIC

; CHECK:        lea %s0, 0
; CHECK-NEXT:   R_VE_LO32 foo
; CHECK-NEXT:   and %s0, %s0, (32)0
; CHECK-NEXT:   lea.sl %s12, (, %s0)
; CHECK-NEXT:   R_VE_HI32 foo
; PIC:        lea %s15, (-24)
; PIC-NEXT:   R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
; PIC-NEXT:   and %s15, %s15, (32)0
; PIC-NEXT:   sic %s16
; PIC-NEXT:   lea.sl %s15, (%s16, %s15)
; PIC-NEXT:   R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
; PIC-NEXT:   lea %s12, (-24)
; PIC-NEXT:   R_VE_PLT_LO32 foo
; PIC-NEXT:   and %s12, %s12, (32)0
; PIC-NEXT:   sic %s16
; PIC-NEXT:   lea.sl %s12, (%s16, %s12)
; PIC-NEXT:   R_VE_PLT_HI32 foo

define i32 @main() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

declare i32 @foo()

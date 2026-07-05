; RUN: not llc -mtriple=xtensa -mattr=+threadptr -relocation-model=pic -filetype=null < %s 2>&1 \
; RUN: | FileCheck -check-prefix=XTENSA-PIC %s
; RUN: not llc -mtriple=xtensa -filetype=null < %s 2>&1 \
; RUN: | FileCheck -check-prefix=XTENSA-NO-THREADPTR %s

; XTENSA-PIC: error: <unknown>:0:0: in function f i32 (): only local-exec and initial-exec TLS mode supported
; XTENSA-PIC: error: <unknown>:0:0: in function f i32 (): PIC relocations are not supported

; XTENSA-NO-THREADPTR: error: <unknown>:0:0: in function f i32 (): only emulated TLS supported

@i = external thread_local global i32

define i32 @f() {
entry:
  %tmp1 = load i32, ptr @i
  ret i32 %tmp1
}

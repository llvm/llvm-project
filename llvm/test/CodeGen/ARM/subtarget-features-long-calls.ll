; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - | FileCheck -check-prefix=NO-OPTION %s
; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=+long-calls | FileCheck -check-prefix=LONGCALL %s
; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - -mattr=-long-calls | FileCheck -check-prefix=NO-LONGCALL %s
; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 | FileCheck -check-prefix=NO-OPTION %s
; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=+long-calls | FileCheck -check-prefix=LONGCALL %s
; RUN: llc -mtriple=thumb-- -mcpu=cortex-a8 -relocation-model=static %s -o - -O0 -mattr=-long-calls | FileCheck -check-prefix=NO-LONGCALL %s
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-a8 -relocation-model=pic %s -o - -O0 -mattr=+long-calls | FileCheck -check-prefix=PIC-O0-LONGCALL %s
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-a8 -relocation-model=pic %s -o - -O1 -mattr=+long-calls | FileCheck -check-prefix=PIC-LONGCALL %s
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-a8 -relocation-model=pic %s -o - -O2 -mattr=+long-calls | FileCheck -check-prefix=PIC-LONGCALL %s

; NO-OPTION-LABEL: {{_?}}caller0
; NO-OPTION: ldr [[R0:r[0-9]+]], [[L0:.*]] 
; NO-OPTION: blx [[R0]]
; NO-OPTION: [[L0]]:
; NO-OPTION: .long {{_?}}callee0

; LONGCALL-LABEL: {{_?}}caller0
; LONGCALL: ldr [[R0:r[0-9]+]], [[L0:.*]]
; LONGCALL: blx [[R0]]
; LONGCALL: [[L0]]:
; LONGCALL: .long {{_?}}callee0

; NO-LONGCALL-LABEL: {{_?}}caller0
; NO-LONGCALL: bl {{_?}}callee0

define i32 @caller0() #0 {
entry:
  tail call void @callee0()
  ret i32 0
}

; NO-OPTION-LABEL: {{_?}}caller1
; NO-OPTION: bl {{_?}}callee0

; LONGCALL-LABEL: {{_?}}caller1
; LONGCALL: ldr [[R0:r[0-9]+]], [[L0:.*]]
; LONGCALL: blx [[R0]]
; LONGCALL: [[L0]]:
; LONGCALL: .long {{_?}}callee0

; NO-LONGCALL-LABEL: {{_?}}caller1
; NO-LONGCALL: bl {{_?}}callee0

define i32 @caller1() {
entry:
  tail call void @callee0()
  ret i32 0
}

declare void @callee0()

; PIC-O0-LONGCALL-LABEL: global_func:
; PIC-O0-LONGCALL:       bx lr

; PIC-LONGCALL-LABEL: global_func:
; PIC-LONGCALL:       bx lr
define void @global_func() {
entry:
  ret void
}

; PIC-O0-LONGCALL-LABEL: test_global:
; PIC-O0-LONGCALL:       push {r11, lr}
; PIC-O0-LONGCALL:       ldr r0, [[GOT_LABEL:.*]]
; PIC-O0-LONGCALL:       ldr r0, [pc, r0]
; PIC-O0-LONGCALL:       blx r0
; PIC-O0-LONGCALL:       pop {r11, pc}
; PIC-O0-LONGCALL:       [[GOT_LABEL]]:
; PIC-O0-LONGCALL:       .long global_func(GOT_PREL)

; PIC-LONGCALL-LABEL: test_global:
; PIC-LONGCALL:       push {r11, lr}
; PIC-LONGCALL:       ldr r0, [[GOT_LABEL:.*]]
; PIC-LONGCALL:       ldr r0, [pc, r0]
; PIC-LONGCALL:       blx r0
; PIC-LONGCALL:       pop {r11, pc}
; PIC-LONGCALL:       [[GOT_LABEL]]:
; PIC-LONGCALL:       .long global_func(GOT_PREL)
define void @test_global() {
entry:
  call void @global_func()
  ret void
}

; PIC-O0-LONGCALL-LABEL: test_memset:
; PIC-O0-LONGCALL:       push {r11, lr}
; PIC-O0-LONGCALL:       ldr r3, [[GOT_LABEL:.*]]
; PIC-O0-LONGCALL:       ldr r3, [pc, r3]
; PIC-O0-LONGCALL:       blx r3
; PIC-O0-LONGCALL:       pop {r11, pc}
; PIC-O0-LONGCALL:       [[GOT_LABEL]]:
; PIC-O0-LONGCALL:       .long memset(GOT_PREL)

; PIC-LONGCALL-LABEL: test_memset:
; PIC-LONGCALL:       push {r11, lr}
; PIC-LONGCALL:       ldr r3, [[MEMSET_LABEL:.*]]
; PIC-LONGCALL:       add r3, pc, r3
; PIC-LONGCALL:       blx r3
; PIC-LONGCALL:       pop {r11, pc}
; PIC-LONGCALL:       [[MEMSET_LABEL]]:
; PIC-LONGCALL:       .long memset
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

define void @test_memset(ptr %dst, i8 %val, i32 %len) {
entry:
  call void @llvm.memset.p0.i32(ptr %dst, i8 %val, i32 %len, i1 false)
  ret void
}

attributes #0 = { "target-features"="+long-calls" }

;RUN: llc -mtriple=sparc -show-mc-encoding < %s | FileCheck %s -check-prefix=V8
;RUN: llc -mtriple=sparc -mattr=v9 < %s | FileCheck %s -check-prefix=V9
;RUN: llc -mtriple=sparc -show-mc-encoding -regalloc=basic < %s | FileCheck %s -check-prefix=V8
;RUN: llc -mtriple=sparc -regalloc=basic -mattr=v9 < %s | FileCheck %s -check-prefix=V9
;RUN: llc -mtriple=sparcv9 < %s | FileCheck %s -check-prefix=SPARC64


define ptr @frameaddr() nounwind readnone {
entry:
;V8-LABEL: frameaddr:
;V8: save %sp, -96, %sp
;V8: ret
;V8: restore %g0, %fp, %o0

;V9-LABEL: frameaddr:
;V9: save %sp, -96, %sp
;V9: ret
;V9: restore %g0, %fp, %o0

;SPARC64-LABEL: frameaddr
;SPARC64:       save %sp, -128, %sp
;SPARC64:       ret
;SPARC64:       restore %fp, 2047, %o0

  %0 = tail call ptr @llvm.frameaddress(i32 0)
  ret ptr %0
}

define ptr @frameaddr2() nounwind readnone {
entry:
;V8-LABEL: frameaddr2:
;V8: ta 3 ! encoding: [0x91,0xd0,0x20,0x03]
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}

;V9-LABEL: frameaddr2:
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}

;SPARC64-LABEL: frameaddr2
;SPARC64: flushw
;SPARC64: ldx [%fp+2159],     %[[R0:[goli][0-7]]]
;SPARC64: ldx [%[[R0]]+2159], %[[R1:[goli][0-7]]]
;SPARC64: ldx [%[[R1]]+2159], %[[R2:[goli][0-7]]]
;SPARC64: ret
;SPARC64: restore %[[R2]], 2047, %o0

  %0 = tail call ptr @llvm.frameaddress(i32 3)
  ret ptr %0
}

declare ptr @llvm.frameaddress(i32) nounwind readnone



define ptr @retaddr() nounwind readnone {
entry:
;V8-LABEL: retaddr:
;V8: mov %o7, {{.+}}

;V9-LABEL: retaddr:
;V9: mov %o7, {{.+}}

;SPARC64-LABEL: retaddr
;SPARC64:       mov %o7, {{.+}}

  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

define ptr @retaddr2() nounwind readnone {
entry:
;V8-LABEL: retaddr2:
;V8: ta 3
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+60], {{.+}}

;V9-LABEL: retaddr2:
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+60], {{.+}}

;SPARC64-LABEL: retaddr2
;SPARC64:       flushw
;SPARC64: ldx [%fp+2159],     %[[R0:[goli][0-7]]]
;SPARC64: ldx [%[[R0]]+2159], %[[R1:[goli][0-7]]]
;SPARC64: ldx [%[[R1]]+2167], {{.+}}

  %0 = tail call ptr @llvm.returnaddress(i32 3)
  ret ptr %0
}

define ptr @retaddr3() nounwind readnone {
entry:
;V8-LABEL: retaddr3:
;V8: ta 3
;V8: ld [%fp+60], {{.+}}

;V9-LABEL: retaddr3:
;V9: flushw
;V9: ld [%fp+60], {{.+}}

;SPARC64-LABEL: retaddr3
;SPARC64:       flushw
;SPARC64: ldx [%fp+2167],     %[[R0:[goli][0-7]]]

  %0 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone

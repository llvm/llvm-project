;RUN: llc -march=sparc < %s | FileCheck %s -check-prefix=V8
;RUN: llc -march=sparc -mattr=v9 < %s | FileCheck %s -check-prefix=V9
;RUN: llc -march=sparc -regalloc=basic < %s | FileCheck %s -check-prefix=V8
;RUN: llc -march=sparc -regalloc=basic -mattr=v9 < %s | FileCheck %s -check-prefix=V9

define i8* @frameaddr() nounwind readnone {
entry:
;V8: frameaddr
;V8: or %g0, %fp, {{.+}}

;V9: frameaddr
;V9: or %g0, %fp, {{.+}}
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @frameaddr2() nounwind readnone {
entry:
;V8: frameaddr2
;V8: ta 3
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}

;V9: frameaddr2
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
  %0 = tail call i8* @llvm.frameaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone



define i8* @retaddr() nounwind readnone {
entry:
;V8: retaddr
;V8: or %g0, %i7, {{.+}}

;V9: retaddr
;V9: or %g0, %i7, {{.+}}
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @retaddr2() nounwind readnone {
entry:
;V8: retaddr2
;V8: ta 3
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+60], {{.+}}

;V9: retaddr2
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+60], {{.+}}
  %0 = tail call i8* @llvm.returnaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

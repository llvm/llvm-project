; RUN: opt -S -globalopt %s | FileCheck %s

; CHECK-LABEL: define void @beth
; CHECK-NEXT:   entry:
; CHECK-NEXT:   ret void
; CHEC-NEXT:  }

@glob = external global i16, align 1

define void @beth() {
entry:
  ret void

notreachable:
  %patatino = select i1 undef, i16* @glob, i16* %patatino
  br label %notreachable
}

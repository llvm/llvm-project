; RUN: llc -mtriple=arm-eabi -pre-RA-sched=source -mattr=+strict-align %s -o - \
; RUN:	| FileCheck %s -check-prefix=EXPANDED

; RUN: llc -mtriple=armv6-apple-darwin -mcpu=cortex-a8 -mattr=-neon,+strict-align -pre-RA-sched=source %s -o - \
; RUN:	| FileCheck %s -check-prefix=EXPANDED

; RUN: llc -mtriple=armv6-apple-darwin -mcpu=cortex-a8 %s -o - \
; RUN:	| FileCheck %s -check-prefix=UNALIGNED

; rdar://7113725
; rdar://12091029

define void @t(i8* nocapture %a, i8* nocapture %b) nounwind {
entry:
; EXPANDED-LABEL: t:
; EXPANDED-DAG: ldrb [[R2:r[0-9]+]]
; EXPANDED-DAG: ldrb [[R3:r[0-9]+]]
; EXPANDED-DAG: ldrb [[R12:r[0-9]+]]
; EXPANDED-DAG: ldrb [[R1:r[0-9]+]]
; EXPANDED-DAG: strb [[R1]]
; EXPANDED-DAG: strb [[R12]]
; EXPANDED-DAG: strb [[R3]]
; EXPANDED-DAG: strb [[R2]]

; UNALIGNED-LABEL: t:
; UNALIGNED: ldr r1
; UNALIGNED: str r1

  %__src1.i = bitcast i8* %b to i32*              ; <i32*> [#uses=1]
  %__dest2.i = bitcast i8* %a to i32*             ; <i32*> [#uses=1]
  %tmp.i = load i32, i32* %__src1.i, align 1           ; <i32> [#uses=1]
  store i32 %tmp.i, i32* %__dest2.i, align 1
  ret void
}

define void @hword(double* %a, double* %b) nounwind {
entry:
; EXPANDED-LABEL: hword:
; EXPANDED-NOT: vld1
; EXPANDED: ldrh
; EXPANDED-NOT: str1
; EXPANDED: strh

; UNALIGNED-LABEL: hword:
; UNALIGNED: vld1.16
; UNALIGNED: vst1.16
  %tmp = load double, double* %a, align 2
  store double %tmp, double* %b, align 2
  ret void
}

define void @byte(double* %a, double* %b) nounwind {
entry:
; EXPANDED-LABEL: byte:
; EXPANDED-NOT: vld1
; EXPANDED: ldrb
; EXPANDED-NOT: str1
; EXPANDED: strb

; UNALIGNED-LABEL: byte:
; UNALIGNED: vld1.8
; UNALIGNED: vst1.8
  %tmp = load double, double* %a, align 1
  store double %tmp, double* %b, align 1
  ret void
}

define void @byte_word_ops(i32* %a, i32* %b) nounwind {
entry:
; EXPANDED-LABEL: byte_word_ops:
; EXPANDED: ldrb
; EXPANDED: strb

; UNALIGNED-LABEL: byte_word_ops:
; UNALIGNED-NOT: ldrb
; UNALIGNED: ldr
; UNALIGNED-NOT: strb
; UNALIGNED: str
  %tmp = load i32, i32* %a, align 1
  store i32 %tmp, i32* %b, align 1
  ret void
}

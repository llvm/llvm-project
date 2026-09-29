; RUN: llc -mtriple=arm-eabi -pre-RA-sched=source -mattr=+strict-align %s -o - \
; RUN:	| FileCheck %s -check-prefix=EXPANDED

; RUN: llc -mtriple=armv6-apple-darwin -mcpu=cortex-a8 -mattr=-neon,+strict-align -pre-RA-sched=source %s -o - \
; RUN:	| FileCheck %s -check-prefix=EXPANDED

; RUN: llc -mtriple=armv6-apple-darwin -mcpu=cortex-a8 %s -o - \
; RUN:	| FileCheck %s -check-prefix=UNALIGNED

; rdar://7113725
; rdar://12091029

define void @t(ptr nocapture %a, ptr nocapture %b) nounwind {
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

  %tmp.i = load i32, ptr %b, align 1           ; <i32> [#uses=1]
  store i32 %tmp.i, ptr %a, align 1
  ret void
}

define void @hword(ptr %a, ptr %b) nounwind {
entry:
; EXPANDED-LABEL: hword:
; EXPANDED-NOT: vld1
; EXPANDED: ldrh
; EXPANDED-NOT: str1
; EXPANDED: strh

; UNALIGNED-LABEL: hword:
; UNALIGNED: vld1.16
; UNALIGNED: vst1.16
  %tmp = load double, ptr %a, align 2
  store double %tmp, ptr %b, align 2
  ret void
}

define void @byte(ptr %a, ptr %b) nounwind {
entry:
; EXPANDED-LABEL: byte:
; EXPANDED-NOT: vld1
; EXPANDED: ldrb
; EXPANDED-NOT: str1
; EXPANDED: strb

; UNALIGNED-LABEL: byte:
; UNALIGNED: vld1.8
; UNALIGNED: vst1.8
  %tmp = load double, ptr %a, align 1
  store double %tmp, ptr %b, align 1
  ret void
}

define void @byte_word_ops(ptr %a, ptr %b) nounwind {
entry:
; EXPANDED-LABEL: byte_word_ops:
; EXPANDED: ldrb
; EXPANDED: strb

; UNALIGNED-LABEL: byte_word_ops:
; UNALIGNED-NOT: ldrb
; UNALIGNED: ldr
; UNALIGNED-NOT: strb
; UNALIGNED: str
  %tmp = load i32, ptr %a, align 1
  store i32 %tmp, ptr %b, align 1
  ret void
}

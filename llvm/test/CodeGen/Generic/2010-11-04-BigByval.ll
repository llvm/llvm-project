; RUN: llc < %s
; PR7170

; The test is intentionally disabled only for the NVPTX target
; (i.e. not for nvptx-registered-target feature) due to excessive runtime.
; Please note, that there are NVPTX special testcases for "byval"
; UNSUPPORTED: nvptx

%big = type [131072 x i8]

declare void @foo(%big* byval(%big) align 1)

define void @bar(%big* byval(%big) align 1 %x) {
  call void @foo(%big* byval(%big) align 1 %x)
  ret void
}

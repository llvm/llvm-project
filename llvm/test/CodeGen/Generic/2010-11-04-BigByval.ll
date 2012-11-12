; RUN: llc < %s
; PR7170

%big = type [131072 x i8]

declare void @foo(%big* byval align 1)

define void @bar(%big* byval align 1 %x) {
  call void @foo(%big* byval align 1 %x)
  ret void
}

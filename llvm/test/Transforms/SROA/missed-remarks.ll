; RUN: opt -passes=sroa -pass-remarks-output=%t.yaml -disable-output < %s
; RUN: FileCheck %s -implicit-check-not=scalar_escapes < %t.yaml

; SROA reports an aggregate allocation whose slices it could not build. An
; escaping use speaks for itself, so it is named by the instruction the remark
; is anchored at; a use slice analysis cannot model also carries a reason. An
; escaping scalar alloca is not reported: there is nothing to split, so it is
; not a missed splitting opportunity.

%struct.S = type { i32, i32 }

declare void @escape(ptr)

; CHECK:      Name:{{ +}}AllocaNotSplit
; CHECK:      Function:{{ +}}aggregate_escapes
; CHECK:      Pointer escapes.

define void @aggregate_escapes() {
entry:
  %s = alloca %struct.S
  call void @escape(ptr %s)
  ret void
}

; CHECK:      Name:{{ +}}AllocaNotSplit
; CHECK:      Function:{{ +}}aggregate_unknown_offset
; CHECK:      Is stored at an offset that is not known.

define void @aggregate_unknown_offset(i64 %n) {
entry:
  %s = alloca %struct.S
  %p = getelementptr i8, ptr %s, i64 %n
  store i32 7, ptr %p
  ret void
}

; CHECK:      Name:{{ +}}AllocaNotSplit
; CHECK:      Function:{{ +}}aggregate_unanalyzable_use
; CHECK:      Has a use that could not be analyzed.

define void @aggregate_unanalyzable_use() {
entry:
  %s = alloca %struct.S
  %v = va_arg ptr %s, i32
  ret void
}

; A scalar alloca has nothing to split, so an escape is not reported.
define void @scalar_escapes() {
entry:
  %x = alloca i32
  call void @escape(ptr %x)
  ret void
}

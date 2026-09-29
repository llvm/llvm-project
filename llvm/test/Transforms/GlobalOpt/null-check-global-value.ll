; RUN: opt -passes=globalopt -S < %s | FileCheck %s

%sometype = type { ptr }

@map = internal unnamed_addr global ptr null, align 8

define void @Init() {
; CHECK-LABEL: @Init(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i1 true, ptr @map.init, align 1
; CHECK-NEXT:    ret void
;
entry:
  %call = tail call noalias nonnull dereferenceable(48) ptr @_Znwm(i64 48)
  store ptr %call, ptr @map, align 8
  ret void
}

define void @Usage() {
; CHECK-LABEL: @Usage(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MAP_INIT_VAL:%.*]] = load i1, ptr @map.init, align 1
; CHECK-NEXT:    [[NOTINIT:%.*]] = xor i1 [[MAP_INIT_VAL]], true
; CHECK-NEXT:    unreachable
;
entry:
  %0 = load ptr, ptr @map, align 8
  %.not = icmp eq ptr %0, null
  unreachable
}

declare ptr @_Znwm(i64)

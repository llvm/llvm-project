; RUN: opt -globalopt -S < %s | FileCheck %s

%sometype = type { i8* }

@map = internal unnamed_addr global %sometype* null, align 8

define void @Init() {
; CHECK-LABEL: @Init(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i1 true, i1* @map.init, align 1
; CHECK-NEXT:    ret void
;
entry:
  %call = tail call noalias nonnull dereferenceable(48) i8* @_Znwm(i64 48)
  store i8* %call, i8** bitcast (%sometype** @map to i8**), align 8
  ret void
}

define void @Usage() {
; CHECK-LABEL: @Usage(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MAP_INIT_VAL:%.*]] = load i1, i1* @map.init, align 1
; CHECK-NEXT:    [[NOTINIT:%.*]] = xor i1 [[MAP_INIT_VAL]], true
; CHECK-NEXT:    unreachable
;
entry:
  %0 = load i8*, i8** bitcast (%sometype** @map to i8**), align 8
  %.not = icmp eq i8* %0, null
  unreachable
}

declare i8* @_Znwm(i64)

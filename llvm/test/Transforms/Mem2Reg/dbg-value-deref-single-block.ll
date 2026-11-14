; RUN: opt < %s -passes='mem2reg' -S | FileCheck %s

;; A deref dbg_value of an alloca promoted by the single-block path describes
;; every later store, until a record of the same variable with an overlapping
;; fragment. The calls are points where the variable can be inspected.

; Each store following the original #dbg_value(deref) is replaced with a
; #dbg_value() record.
define i32 @before_both(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @before_both(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %a, ptr %x
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A store before the original debug record doesn't receive a new record.
define i32 @between(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @between(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
  %x = alloca i32
  store i32 %a, ptr %x
  call void @step()
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A store after an overlapping debug record with a different value doesn't
; receive a new record.
define i32 @overridden(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @overridden(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 0, {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %a, ptr %x
  call void @step()
    #dbg_value(i32 0, !20, !DIExpression(), !30)
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A new #dbg_value(deref) after an override applies to later stores again.
define i32 @redescribed(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @redescribed(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 0, {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %a, ptr %x
  call void @step()
    #dbg_value(i32 0, !20, !DIExpression(), !30)
  call void @step()
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A disjoint fragment is kept and doesn't disturb propagation.
define i32 @disjoint_fragment(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @disjoint_fragment(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 1, {{.*}}, !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_value(ptr %x, !21, !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32), !30)
  store i32 %a, ptr %x
  call void @step()
    #dbg_value(i32 1, !21, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !30)
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A store after a record of a fragment contained in the original record's
; fragment doesn't receive a new record.
define i32 @contained_fragment(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @contained_fragment(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i16 2, {{.*}}, !DIExpression(DW_OP_LLVM_fragment, 0, 16),
; CHECK-NEXT:    call
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_value(ptr %x, !21, !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32), !30)
  store i32 %a, ptr %x
  call void @step()
    #dbg_value(i16 2, !21, !DIExpression(DW_OP_LLVM_fragment, 0, 16), !30)
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A #dbg_declare_value(deref) holds for the whole function like a #dbg_declare,
; so every store receives a new record regardless of its position.
define i32 @declare_value_deref(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @declare_value_deref(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
  %x = alloca i32
  store i32 %a, ptr %x
  call void @step()
    #dbg_declare_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; A #dbg_declare_value without a deref describes the address itself, which is
; gone after promotion.
define i32 @declare_value(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @declare_value(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_declare_value(ptr poison, {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    call
  %x = alloca i32
    #dbg_declare_value(ptr %x, !22, !DIExpression(), !30)
  store i32 %a, ptr %x
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  %v = load i32, ptr %x
  ret i32 %v
}

; Lifetime markers shouldn't affect the debug record.
define void @lifetime(ptr %v) {
; CHECK-LABEL: define void @lifetime(
; CHECK-SAME:    ptr [[V:%.*]])
; CHECK-NEXT:    #dbg_value(ptr [[V]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    ret void
  %a = alloca ptr
    #dbg_value(ptr %a, !22, !DIExpression(DW_OP_deref), !30)
  call void @llvm.lifetime.start.p0(ptr %a)
  store ptr %v, ptr %a
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}

; A variable moving to another alloca ends the first alloca's record, even if
; the second alloca is promoted first.
define void @moved(i32 %a1, i32 %a2, i32 %b0, i32 %b1) {
; CHECK-LABEL: define void @moved(
; CHECK-SAME:    i32 [[A1:%.*]], i32 [[A2:%.*]], i32 [[B0:%.*]], i32 [[B1:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A1]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B1]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    call
; CHECK-NEXT:    ret void
  %b = alloca i32
  %a = alloca i32
    #dbg_value(ptr %a, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %a1, ptr %a
  call void @step()
  store i32 %b0, ptr %b
  store i32 %b1, ptr %b
    #dbg_value(ptr %b, !20, !DIExpression(DW_OP_deref), !30)
  call void @step()
  store i32 %a2, ptr %a
  call void @step()
  ret void
}

; Same, with the first alloca promoted first.
define void @moved_swapped(i32 %a1, i32 %a2, i32 %b0, i32 %b1) {
; CHECK-LABEL: define void @moved_swapped(
; CHECK-SAME:    i32 [[A1:%.*]], i32 [[A2:%.*]], i32 [[B0:%.*]], i32 [[B1:%.*]])
; CHECK-NEXT:    #dbg_value(i32 [[A1]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B1]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    call
; CHECK-NEXT:    ret void
  %a = alloca i32
  %b = alloca i32
    #dbg_value(ptr %a, !20, !DIExpression(DW_OP_deref), !30)
  store i32 %a1, ptr %a
  call void @step()
  store i32 %b0, ptr %b
  store i32 %b1, ptr %b
    #dbg_value(ptr %b, !20, !DIExpression(DW_OP_deref), !30)
  call void @step()
  store i32 %a2, ptr %a
  call void @step()
  ret void
}

; A record with no store before it is removed.
define void @uninitialized(i32 %a, i32 %b) {
; CHECK-LABEL: define void @uninitialized(
; CHECK-SAME:    i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    #dbg_value(i32 0, {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[A]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(i32 [[B]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    ret void
  %x = alloca i32
    #dbg_value(i32 0, !20, !DIExpression(), !30)
    #dbg_value(ptr %x, !20, !DIExpression(DW_OP_deref), !30)
  call void @step()
  store i32 %a, ptr %x
  call void @step()
  store i32 %b, ptr %x
  call void @step()
  ret void
}

; The lifetime marker doesn't create a poison record before the first step.
define void @lifetime_uninitialized(ptr %v) {
; CHECK-LABEL: define void @lifetime_uninitialized(
; CHECK-SAME:    ptr [[V:%.*]])
; CHECK-NEXT:    #dbg_value(ptr null, {{.*}}, !DIExpression(),
; CHECK-NEXT:    call
; CHECK-NEXT:    #dbg_value(ptr [[V]], {{.*}}, !DIExpression(),
; CHECK-NEXT:    ret void
  %a = alloca ptr
    #dbg_value(ptr null, !22, !DIExpression(), !30)
  call void @llvm.lifetime.start.p0(ptr %a)
    #dbg_value(ptr %a, !22, !DIExpression(DW_OP_deref), !30)
  call void @step()
  store ptr %v, ptr %a
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}


declare void @step()

!llvm.module.flags = !{!1, !2}
!llvm.dbg.cu = !{!7}

!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang", emissionKind: FullDebug)
!8 = !DIFile(filename: "test.ll", directory: "")
!9 = !DISubroutineType(types: !{null})
!10 = distinct !DISubprogram(name: "f", scope: !8, file: !8, line: 1, type: !9, unit: !7)
!20 = !DILocalVariable(name: "x", scope: !10, file: !8, line: 1, type: !3)
!21 = !DILocalVariable(name: "y", scope: !10, file: !8, line: 1, type: !4)
!22 = !DILocalVariable(name: "p", scope: !10, file: !8, line: 1, type: !5)
!30 = !DILocation(line: 1, column: 1, scope: !10)

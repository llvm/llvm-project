; RUN: opt < %s -passes=instcombine -S | FileCheck --match-full-lines %s

; Test cases to make sure !annotation metadata is preserved, if possible.
; Currently we fail to preserve !annotation metadata in many cases.

; Make sure !annotation metadata is added to new instructions, if the source
; instruction has !annotation metadata.
define i1 @fold_to_new_instruction(ptr %a, ptr %b) {
; CHECK-LABEL: define {{.+}} @fold_to_new_instruction({{.+}}
; CHECK-NEXT:    [[C:%.*]] = icmp uge ptr [[A:%.*]], [[B:%[a-z]*]], !annotation [[ANN:![0-9]+]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %c = icmp uge ptr %a, %b, !annotation !0
  ret i1 %c
}

; Make sure !annotation is not added to new instructions if the source
; instruction does not have it (even if some folded operands do have
; !annotation).
define i1 @fold_to_new_instruction2(ptr %a, ptr %b) {
; CHECK-LABEL: define {{.+}} @fold_to_new_instruction2({{.+}}
; CHECK-NEXT:    [[C:%.*]] = icmp uge ptr [[A:%.*]], [[B:%[a-z]+]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %c = icmp uge ptr %a, %b
  ret i1 %c
}

; Make sure !annotation metadata is *not* added if we replace an instruction
; with !annotation with an existing one without.
define i32 @do_not_add_annotation_to_existing_instr(i32 %a, i32 %b) {
; CHECK-LABEL: define {{.+}} @do_not_add_annotation_to_existing_instr({{.+}}
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[A:%.*]], [[B:%[a-z]+]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %add = add i32 %a, %b
  %res = add i32 0, %add, !annotation !0
  ret i32 %res
}

; memcpy can be expanded inline with load/store. Verify that we keep the
; !annotation metadata.

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

define void @copy_1_byte(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @copy_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0.p0.i32(ptr %d, ptr %s, i32 1, i1 false), !annotation !0
  ret void
}

declare ptr @memcpy(ptr noalias returned, ptr noalias nocapture readonly, i64) nofree nounwind

define void @libcallcopy_1_byte(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @libcallcopy_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @memcpy(ptr %d, ptr %s, i64 1), !annotation !0
  ret void
}

declare ptr @__memcpy_chk(ptr, ptr, i64, i64) nofree nounwind

define void @libcallcopy_1_byte_chk(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @libcallcopy_1_byte_chk({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @__memcpy_chk(ptr %d, ptr %s, i64 1, i64 1), !annotation !0
  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) nounwind

define void @move_1_byte(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @move_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memmove.p0.p0.i32(ptr %d, ptr %s, i32 1, i1 false), !annotation !0
  ret void
}

declare ptr @memmove(ptr returned, ptr nocapture readonly, i64) nofree nounwind

define void @libcallmove_1_byte(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @libcallmove_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @memmove(ptr %d, ptr %s, i64 1), !annotation !0
  ret void
}

declare ptr @__memmove_chk(ptr, ptr, i64, i64) nofree nounwind

define void @libcallmove_1_byte_chk(ptr %d, ptr %s) {
; CHECK-LABEL: define {{.+}} @libcallmove_1_byte_chk({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @__memmove_chk(ptr %d, ptr %s, i64 1, i64 1), !annotation !0
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) argmemonly nounwind

define void @set_1_byte(ptr %d) {
; CHECK-LABEL: define {{.+}} @set_1_byte({{.+}}
; CHECK-NEXT:    store i8 1, ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memset.p0.i32(ptr %d, i8 1, i32 1, i1 false), !annotation !0
  ret void
}

declare ptr @memset(ptr, i32, i64) nofree

define void @libcall_set_1_byte(ptr %d) {
; CHECK-LABEL: define {{.+}} @libcall_set_1_byte({{.+}}
; CHECK-NEXT:    store i8 1, ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @memset(ptr %d, i32 1, i64 1), !annotation !0
  ret void
}

declare ptr @__memset_chk(ptr, i32, i64, i64) nofree

define void @libcall_set_1_byte_chk(ptr %d) {
; CHECK-LABEL: define {{.+}} @libcall_set_1_byte_chk({{.+}}
; CHECK-NEXT:    store i8 1, ptr [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call ptr @__memset_chk(ptr %d, i32 1, i64 1, i64 1), !annotation !0
  ret void
}

!0 = !{ !"auto-init" }

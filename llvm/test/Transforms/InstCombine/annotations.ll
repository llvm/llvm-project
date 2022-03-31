; RUN: opt < %s -passes=instcombine -S | FileCheck --match-full-lines %s

; Test cases to make sure !annotation metadata is preserved, if possible.
; Currently we fail to preserve !annotation metadata in many cases.

; Make sure !annotation metadata is added to new instructions, if the source
; instruction has !annotation metadata.
define i1 @fold_to_new_instruction(i8* %a, i8* %b) {
; CHECK-LABEL: define {{.+}} @fold_to_new_instruction({{.+}}
; CHECK-NEXT:    [[C:%.*]] = icmp uge i8* [[A:%.*]], [[B:%[a-z]*]], !annotation [[ANN:![0-9]+]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a.c = bitcast i8* %a to i32*, !annotation !0
  %b.c = bitcast i8* %b to i32*, !annotation !0
  %c = icmp uge i32* %a.c, %b.c, !annotation !0
  ret i1 %c
}

; Make sure !annotation is not added to new instructions if the source
; instruction does not have it (even if some folded operands do have
; !annotation).
define i1 @fold_to_new_instruction2(i8* %a, i8* %b) {
; CHECK-LABEL: define {{.+}} @fold_to_new_instruction2({{.+}}
; CHECK-NEXT:    [[C:%.*]] = icmp uge i8* [[A:%.*]], [[B:%[a-z]+]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %a.c = bitcast i8* %a to i32*, !annotation !0
  %b.c = bitcast i8* %b to i32*, !annotation !0
  %c = icmp uge i32* %a.c, %b.c
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

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @copy_1_byte(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @copy_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d, i8* %s, i32 1, i1 false), !annotation !0
  ret void
}

declare i8* @memcpy(i8* noalias returned, i8* noalias nocapture readonly, i64) nofree nounwind

define void @libcallcopy_1_byte(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @libcallcopy_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @memcpy(i8* %d, i8* %s, i64 1), !annotation !0
  ret void
}

declare i8* @__memcpy_chk(i8*, i8*, i64, i64) nofree nounwind

define void @libcallcopy_1_byte_chk(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @libcallcopy_1_byte_chk({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @__memcpy_chk(i8* %d, i8* %s, i64 1, i64 1), !annotation !0
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) nounwind

define void @move_1_byte(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @move_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %d, i8* %s, i32 1, i1 false), !annotation !0
  ret void
}

declare i8* @memmove(i8* returned, i8* nocapture readonly, i64) nofree nounwind

define void @libcallmove_1_byte(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @libcallmove_1_byte({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @memmove(i8* %d, i8* %s, i64 1), !annotation !0
  ret void
}

declare i8* @__memmove_chk(i8*, i8*, i64, i64) nofree nounwind

define void @libcallmove_1_byte_chk(i8* %d, i8* %s) {
; CHECK-LABEL: define {{.+}} @libcallmove_1_byte_chk({{.+}}
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* [[S:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @__memmove_chk(i8* %d, i8* %s, i64 1, i64 1), !annotation !0
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) argmemonly nounwind

define void @set_1_byte(i8* %d) {
; CHECK-LABEL: define {{.+}} @set_1_byte({{.+}}
; CHECK-NEXT:    store i8 1, i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memset.p0i8.i32(i8* %d, i8 1, i32 1, i1 false), !annotation !0
  ret void
}

declare i8* @memset(i8*, i32, i64) nofree

define void @libcall_set_1_byte(i8* %d) {
; CHECK-LABEL: define {{.+}} @libcall_set_1_byte({{.+}}
; CHECK-NEXT:    store i8 1, i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @memset(i8* %d, i32 1, i64 1), !annotation !0
  ret void
}

declare i8* @__memset_chk(i8*, i32, i64, i64) nofree

define void @libcall_set_1_byte_chk(i8* %d) {
; CHECK-LABEL: define {{.+}} @libcall_set_1_byte_chk({{.+}}
; CHECK-NEXT:    store i8 1, i8* [[D:%.*]], align 1, !annotation [[ANN]]
; CHECK-NEXT:    ret void
;
  call i8* @__memset_chk(i8* %d, i32 1, i64 1, i64 1), !annotation !0
  ret void
}

!0 = !{ !"auto-init" }

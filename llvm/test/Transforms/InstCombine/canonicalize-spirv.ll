; RUN: opt -S -passes=instcombine < %s | FileCheck --check-prefix=BASE %s
; RUN: opt -S -mtriple=spirv-- -passes=instcombine < %s | FileCheck --check-prefix=SPIR %s

define float @foo(ptr %x) {
; BASE-LABEL: define float @foo(
; BASE-SAME: ptr [[X:%.*]]) {
; BASE-NEXT:  entry:
; BASE-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[X]], i64 4
; BASE-NEXT:    [[TMP1:%.*]] = load float, ptr [[TMP0]], align 4
; BASE-NEXT:    ret float [[TMP1]]
;
; SPIR-LABEL: define float @foo(
; SPIR-SAME: ptr [[X:%.*]]) {
; SPIR-NEXT:  entry:
; SPIR-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw <4 x float>, ptr [[X]], i64 0, i64 1
; SPIR-NEXT:    [[TMP1:%.*]] = load float, ptr [[TMP0]], align 4
; SPIR-NEXT:    ret float [[TMP1]]
;
entry:
  %3 = getelementptr inbounds nuw <4 x float>, ptr %x, i32 0, i64 1
  %4 = load float, ptr %3, align 4
  ret float %4
}

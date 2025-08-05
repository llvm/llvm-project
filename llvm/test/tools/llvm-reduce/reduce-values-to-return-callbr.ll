; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions-to-return --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

@gv = global i32 0, align 4

; INTERESTING-LABEL: @callbr0(
; INTERESTING: %load0 = load i32, ptr %arg0
; INTERESTING: store i32 %load0, ptr @gv

; RESULT-LABEL: define void @callbr0(ptr %arg0) {
; RESULT: %load0 = load i32, ptr %arg0, align 4
; RESULT-NEXT: %callbr = callbr i32 asm
define void @callbr0(ptr %arg0) {
entry:
  %load0 = load i32, ptr %arg0
  %callbr = callbr i32 asm "", "=r,r,!i,!i"(i32 %load0)
              to label %one [label %two, label %three]
one:
  store i32 %load0, ptr @gv
  ret void

two:
  store i32 %load0, ptr @gv
  ret void

three:
  store i32 %load0, ptr @gv
  ret void
}

; INTERESTING-LABEL: @callbr1(
; INTERESTING: %load0 = load i32, ptr %arg0

; RESULT-LABEL: define i32 @callbr1(ptr %arg0) {
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg0
; RESULT-NEXT: ret i32 %load0
define void @callbr1(ptr %arg0) {
entry:
  %load0 = load i32, ptr %arg0
  %callbr = callbr i32 asm "", "=r,r,!i,!i"(i32 %load0)
              to label %one [label %two, label %three]
one:
  store i32 %load0, ptr @gv
  ret void

two:
  store i32 %load0, ptr @gv
  ret void

three:
  store i32 %load0, ptr @gv
  ret void
}

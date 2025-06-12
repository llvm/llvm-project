; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

@a = dso_local global i8 0, align 1
@b = dso_local global i16 0, align 2

; INTERESTING-LABEL: define void @c(
; INTERESTING: sext
; INTERESTING: icmp

; REDUCED: define void @c(ptr %a, i8 %ld0, ptr %b, i16 %ld1, i32 %conv, i32 %conv1, i1 %cmp, i32 %conv2)
; REDUCED: call void @c(i32 noundef signext %conv2)
define void @c() {
entry:
  %ld0 = load i8, ptr @a, align 1
  %conv = zext i8 %ld0 to i32
  %ld1 = load i16, ptr @b, align 2
  %conv1 = sext i16 %ld1 to i32
  %cmp = icmp sge i32 %conv, %conv1
  %conv2 = zext i1 %cmp to i32
  call void @c(i32 noundef signext %conv2)
  ret void
}


; RUN: llc < %s -mtriple=x86_64-unknown-unknown-eabi-elf | FileCheck %s

; CHECK-LABEL: conv2d
define dso_local void @conv2d() {
.preheader:
  br label %0

0:                                                ; preds = %0, %.preheader
  %1 = phi [4 x <7 x half>] [ zeroinitializer, %.preheader ], [ %4, %0 ]
  %2 = extractvalue [4 x <7 x half>] %1, 0
  %3 = extractvalue [4 x <7 x half>] %1, 1
  %4 = insertvalue [4 x <7 x half>] poison, <7 x half> poison, 3
  br label %0
}

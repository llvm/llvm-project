;; This file tests specifying the debug info filename in the basic block sections profile.
;; Specify the right filename.
; RUN: echo '!test M=/path/to/dir/test_filename' > %t1
; RUN: echo '!!0' >> %t1
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t1 | FileCheck %s -check-prefix=RIGHT-MODULE
;; Specify no filename and verify that the profile is ingested.
; RUN: echo '!test' > %t2
; RUN: echo '!!0' >> %t2
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t2  | FileCheck %s -check-prefix=NO-MODULE
;; Specify wrong filenames and verify that the profile is not ingested.
; RUN: echo '!test M=test_filename' > %t3
; RUN: echo '!!0' >> %t3
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t3  | FileCheck %s -check-prefix=WRONG-MODULE
; RUN: echo '!test M=./path/to/dir/test_filename' > %t4
; RUN: echo '!!0' >> %t4
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t4  | FileCheck %s -check-prefix=WRONG-MODULE
;; Version 1 profile.
;; Specify the right filename.
; RUN: echo 'v1' > %t5
; RUN: echo 'm /path/to/dir/test_filename' >> %t5
; RUN: echo 'f test' >> %t5
; RUN: echo 'c 0' >> %t5
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t5 | FileCheck %s -check-prefix=RIGHT-MODULE
;; Specify no filename and verify that the profile is ingested.
; RUN: echo 'v1' > %t6
; RUN: echo 'f test' >> %t6
; RUN: echo 'c 0' >> %t6
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t6  | FileCheck %s -check-prefix=NO-MODULE
;; Specify wrong filenames and verify that the profile is not ingested.
; RUN: echo 'v1' > %t7
; RUN: echo 'm test_filename' >> %t7
; RUN: echo 'f test' >> %t7
; RUN: echo 'c 0' >> %t7
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t7  | FileCheck %s -check-prefix=WRONG-MODULE
; RUN: echo 'v1' > %t8
; RUN: echo 'm ./path/to/dir/test_filename' >> %t8
; RUN: echo 'f test' >> %t8
; RUN: echo 'c 0' >> %t8
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t8  | FileCheck %s -check-prefix=WRONG-MODULE


define dso_local i32 @test(i32 noundef %0) #0 !dbg !10 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  %4 = load i32, ptr %3, align 4
  %5 = icmp slt i32 %4, 0
  br i1 %5, label %6, label %7
6:                                                ; preds = %1
  store i32 -1, ptr %2, align 4
  ret i32 0
7:
  ret i32 1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "/path/to/dir/test_filename", directory: "/path/to/dir")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "test", scope: !1, unit: !0)

;; Check that the split section is created when using the correct module name, or no module name.
;
; RIGHT-MODULE: .section        .text.split.test,"ax",@progbits
; NO-MODULE: .section        .text.split.test,"ax",@progbits
; WRONG-MODULE-NOT: .section    .text.split.test,"ax",@progbits

;; This file tests various ways of specifying the debug info filename in the basic block sections profile.
;; Specifying correct filenames.
; RUN: echo '!test1 M=./test_dirname1/test_filename1' > %t1
; RUN: echo '!!0' >> %t1
; RUN: echo '!test2 M=.//test_filename2' >> %t1
; RUN: echo '!!0' >> %t1
; RUN: echo '!test3 M=test_filename3' >> %t1
; RUN: echo '!!0' >> %t1
; RUN: echo '!test4 M=/test_dirname4/test_filename4' >> %t1
; RUN: echo '!!0' >> %t1
; RUN: echo '!test5' >> %t1
; RUN: echo '!!0' >> %t1
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t1  | FileCheck %s -check-prefix=RIGHT-MODULE
;; Specifying wrong filenames.
; RUN: echo '!test1 M=/test_dirname/test_filename1' > %t2
; RUN: echo '!!0' >> %t2
; RUN: echo '!test1 M=../test_filename1' >> %t2
; RUN: echo '!!0' >> %t2
; RUN: echo '!test2 M=.test_filename2' >> %t2
; RUN: echo '!!0' >> %t2
; RUN: echo '!test4 M=./test_dirname4/test_filename4' >> %t2
; RUN: echo '!!0' >> %t2
; RUN: echo '!test5 M=any_filename' >> %t1
; RUN: echo '!!0' >> %t1
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t2  | FileCheck %s -check-prefix=WRONG-MODULE

define dso_local i32 @test1(i32 noundef %0) #0 !dbg !10 {
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

define dso_local i32 @test2(i32 noundef %0) #0 !dbg !11 {
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

define dso_local i32 @test3(i32 noundef %0) #0 !dbg !12 {
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

define dso_local i32 @test4(i32 noundef %0) #0 !dbg !13 {
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

define dso_local i32 @test5(i32 noundef %0) #0 {
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

!llvm.dbg.cu = !{!0, !1, !2, !3}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7)
!4 = !DIFile(filename: "test_dirname1/test_filename1", directory: "test_dirname1")
!5 = !DIFile(filename: "test_filename2", directory: "")
!6 = !DIFile(filename: "./test_filename3", directory: ".")
!7 = !DIFile(filename: "/test_dirname4/test_filename4", directory: "/test_dirname4")
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "test1", scope: !4, unit: !0)
!11 = distinct !DISubprogram(name: "test2", scope: !5, unit: !1)
!12 = distinct !DISubprogram(name: "test3", scope: !6, unit: !2)
!13 = distinct !DISubprogram(name: "test4", scope: !7, unit: !3)

;; Check that the split section is created when using the correct module name, or no module name.
;
; RIGHT-MODULE: .section        .text.split.test1,"ax",@progbits
; RIGHT-MODULE: .section        .text.split.test2,"ax",@progbits
; RIGHT-MODULE: .section        .text.split.test3,"ax",@progbits
; RIGHT-MODULE: .section        .text.split.test4,"ax",@progbits
; RIGHT-MODULE: .section        .text.split.test5,"ax",@progbits
; WRONG-MODULE-NOT: .section        .text.split.test{{[1-5]+}},"ax",@progbits

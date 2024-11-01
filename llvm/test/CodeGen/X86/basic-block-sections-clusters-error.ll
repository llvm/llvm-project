; BB cluster sections error handling
; RUN: echo '!dummy1' > %t1
; RUN: echo '!!1 4' >> %t1
; RUN: echo '!!1' >> %t1
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR1
; CHECK-ERROR1: LLVM ERROR: Invalid profile {{.*}} at line 3: Duplicate basic block id found '1'.
; RUN: echo '!dummy1' > %t2
; RUN: echo '!!4 0' >> %t2
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR2
; CHECK-ERROR2: LLVM ERROR: Invalid profile {{.*}} at line 2: Entry BB (0) does not begin a cluster.
; RUN: echo '!dummy1' > %t3
; RUN: echo '!!-1' >> %t3
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR3
; CHECK-ERROR3: LLVM ERROR: Invalid profile {{.*}} at line 2: Unsigned integer expected: '-1'.
; RUN: echo '!dummy1 /path/to/filename' > %t4
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR4
; CHECK-ERROR4: LLVM ERROR: Invalid profile {{.*}} at line 1: Unknown string found: '/path/to/filename'.
; RUN: echo '!dummy2 M=test_dir/test_file' > %t5
; RUN: echo '!dummy2 M=test_dir/test_file' >> %t5
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t5 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR5
; CHECK-ERROR5: LLVM ERROR: Invalid profile {{.*}} at line 2: Duplicate profile for function 'dummy2'.
; RUN: echo '!dummy1 M=' >> %t6
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR6
; CHECK-ERROR6: LLVM ERROR: Invalid profile {{.*}} at line 1: Empty module name specifier.

define i32 @dummy1(i32 %x, i32 %y, i32 %z) {
  entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}

define i32 @dummy2(i32 %x, i32 %y, i32 %z) !dbg !4 {
  entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test_dir/test_file", directory: "test_dir")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "dummy1", scope: !1, unit: !0)



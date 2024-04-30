;; BB cluster sections error handling
;; Error handling for version 0:
; RUN: echo '!dummy1' > %t1
; RUN: echo '!!1 4' >> %t1
; RUN: echo '!!1' >> %t1
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR1
; CHECK-ERROR1: LLVM ERROR: invalid profile {{.*}} at line 3: duplicate basic block id found '1'
; RUN: echo '!dummy1' > %t3
; RUN: echo '!!-1' >> %t3
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR3
; CHECK-ERROR3: LLVM ERROR: invalid profile {{.*}} at line 2: unsigned integer expected: '-1'
; RUN: echo '!dummy1 /path/to/filename' > %t4
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR4
; CHECK-ERROR4: LLVM ERROR: invalid profile {{.*}} at line 1: unknown string found: '/path/to/filename'
; RUN: echo '!dummy2 M=test_dir/test_file' > %t5
; RUN: echo '!dummy2 M=test_dir/test_file' >> %t5
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t5 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR5
; CHECK-ERROR5: LLVM ERROR: invalid profile {{.*}} at line 2: duplicate profile for function 'dummy2'
; RUN: echo '!dummy1 M=' > %t6
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR6
; CHECK-ERROR6: LLVM ERROR: invalid profile {{.*}} at line 1: empty module name specifier
;;
;; Error handling for version 1:
; RUN: echo 'v2' > %t7
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t7 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR7
; CHECK-ERROR7: LLVM ERROR: invalid profile {{.*}} at line 1: invalid profile version: 2
; RUN: echo 'v1' > %t8
; RUN: echo '!dummy1' >> %t8
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t8 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR8
; CHECK-ERROR8: LLVM ERROR: invalid profile {{.*}} at line 2: invalid specifier: '!'
; RUN: echo 'v1' > %t9
; RUN: echo 'm dummy1/module1 dummy1/module2' >> %t9
; RUN: echo 'f dummy1' >> %t9
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t9 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR9
; CHECK-ERROR9: LLVM ERROR: invalid profile {{.*}} at line 2: invalid module name value: 'dummy1/module1 dummy1/module2'
;;
;; Error handling for version 1, cloning paths.
; RUN: echo 'v1' > %t10
; RUN: echo 'f dummy1' >> %t10
; RUN: echo 'c 0 1.1.1' >> %t10
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t10 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR10
; CHECK-ERROR10: LLVM ERROR: invalid profile {{.*}} at line 3: unable to parse basic block id: '1.1.1'
; RUN: echo 'v1' > %t11
; RUN: echo 'f dummy1' >> %t11
; RUN: echo 'c 0 1.a' >> %t11
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t11 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR11
; CHECK-ERROR11: LLVM ERROR: invalid profile {{.*}} at line 3: unable to parse clone id: 'a'
; RUN: echo 'v1' > %t12
; RUN: echo 'f dummy1' >> %t12
; RUN: echo 'c 0 1' >> %t12
; RUN: echo 'p 1 2.1' >> %t12
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t12 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR12
; CHECK-ERROR12: LLVM ERROR: invalid profile {{.*}} at line 4: unsigned integer expected: '2.1'
; RUN: echo 'v1' > %t13
; RUN: echo 'f dummy1' >> %t13
; RUN: echo 'c 0 1' >> %t13
; RUN: echo 'p 1 2 3 2' >> %t13
; RUN: not --crash llc < %s -O0 -mtriple=x86_64 -function-sections -basic-block-sections=%t13 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR13
; CHECK-ERROR13: LLVM ERROR: invalid profile {{.*}} at line 4: duplicate cloned block in path: '2'

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


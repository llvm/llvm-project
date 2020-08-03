; BB cluster sections error handling
; RUN: echo '!f' > %t1
; RUN: echo '!!1 4' >> %t1
; RUN: echo '!!1' >> %t1
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR1
; CHECK-ERROR1: LLVM ERROR: Invalid profile {{.*}} at line 3: Duplicate basic block id found '1'.
; RUN: echo '!f' > %t2
; RUN: echo '!!4 0' >> %t2
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR2
; CHECK-ERROR2: LLVM ERROR: Invalid profile {{.*}} at line 2: Entry BB (0) does not begin a cluster.
; RUN: echo '!f' > %t3
; RUN: echo '!!-1' >> %t3
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR3
; CHECK-ERROR3: LLVM ERROR: Invalid profile {{.*}} at line 2: Unsigned integer expected: '-1'.
; RUN: echo '!!1' > %t4
; RUN: echo '!f' >> %t4
; RUN: not --crash llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR4
; CHECK-ERROR4: LLVM ERROR: Invalid profile {{.*}} at line 1: Cluster list does not follow a function name specifier.

define i32 @dummy(i32 %x, i32 %y, i32 %z) {
  entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}

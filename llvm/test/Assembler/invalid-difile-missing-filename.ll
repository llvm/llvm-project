; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:30: error: missing required field 'filename'
!0 = !DIFile(directory: "dir")

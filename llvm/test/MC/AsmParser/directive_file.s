# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: llvm-mc -triple i386-unknown-unknown %s -filetype=null

        .file "hello"
        .file 1 "worl\144"   # "\144" is "d"
        .file 2 "directory" "file"

# CHECK: .file "hello"
# CHECK: .file 1 "world"
# CHECK: .file 2 "directory" "file"

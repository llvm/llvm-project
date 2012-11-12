# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .file "hello"
        .file 1 "world"
        .file 2 "directory" "file"

# CHECK: .file "hello"
# CHECK: .file 1 "world"
# CHECK: .file 2 "directory" "file"

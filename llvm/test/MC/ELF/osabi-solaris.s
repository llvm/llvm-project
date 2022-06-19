# RUN: llvm-mc -filetype obj -triple amd64-solaris %s | llvm-readobj -hS - | FileCheck %s
# CHECK: OS/ABI: Solaris

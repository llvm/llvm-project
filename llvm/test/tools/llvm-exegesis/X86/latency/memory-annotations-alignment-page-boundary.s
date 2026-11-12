# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Test that we error out if the requested address is not aligned to a page
# boundary. Here we test out mapping at 2^12+4, which is off from a page
# boundary by four bytes.

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess 2>&1 | FileCheck %s

# LLVM-EXEGESIS-MEM-DEF test1 4096 414D47
# LLVM-EXEGESIS-MEM-MAP test1 65540

nop

# CHECK: invalid comment 'LLVM-EXEGESIS-MEM-MAP  test1 65540', expected <ADDRESS> to be a multiple of the platform page size. 

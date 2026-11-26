# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -instruction-tables < %s | FileCheck %s

; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck %s
; CHECK: .file "foobar"

source_filename = "foobar"

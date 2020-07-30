; RUN: not opt -dce --passes=inline %s 2>&1 | FileCheck %s
; CHECK: Cannot specify passes via both -foo-pass and --passes=foo-pass

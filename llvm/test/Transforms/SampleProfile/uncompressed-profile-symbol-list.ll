; Append inline.prof with profile symbol list and save it without compression.
; RUN: llvm-profdata merge --sample --prof-sym-list %S/Inputs/profile-symbol-list.text --extbinary %S/Inputs/inline.prof -o %t.profdata
; RUN: opt < %S/Inputs/profile-symbol-list.ll -passes=sample-profile -profile-accurate-for-symsinlist -sample-profile-file=%t.profdata -S | FileCheck %S/Inputs/profile-symbol-list.ll

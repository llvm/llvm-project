; REQUIRES: zlib
; Append inline.prof with profile symbol list and save it after compression.
; RUN: llvm-profdata merge --sample --prof-sym-list=%S/Inputs/profile-symbol-list.text --compress-all-sections=true --extbinary %S/Inputs/inline.prof --output=%t.profdata
; RUN: opt < %S/Inputs/profile-symbol-list.ll -sample-profile -profile-accurate-for-symsinlist -sample-profile-file=%t.profdata -S | FileCheck %S/Inputs/profile-symbol-list.ll
; RUN: opt < %S/Inputs/profile-symbol-list.ll -passes=sample-profile -profile-accurate-for-symsinlist -sample-profile-file=%t.profdata -S | FileCheck %S/Inputs/profile-symbol-list.ll

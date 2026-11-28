; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare dso_local dllimport void @fun()
; CHECK: error: dso_location and DLL-StorageClass mismatch

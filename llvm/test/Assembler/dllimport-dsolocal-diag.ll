; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare dso_local dllimport void @fun()
; CHECK: error: dso_location and DLL-StorageClass mismatch

; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:51: error: missing required field 'scope'
!3 = !DIImportedEntity(tag: DW_TAG_imported_module)

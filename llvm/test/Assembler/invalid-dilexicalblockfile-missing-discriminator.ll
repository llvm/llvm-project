; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:36: error: missing required field 'discriminator'
!0 = !DILexicalBlockFile(scope: !{})

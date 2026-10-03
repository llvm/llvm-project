; RUN: cat %S/SemanticSignatures-vs.ll %s > %t.ll
; RUN: llc -filetype=obj -dx-embed-debug %t.ll -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %S/SemanticSignatures-vs.ll --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.dxil.bc %t.dxbc
; RUN: llvm-objcopy --dump-section=ILDB=%t.ildb.bc %t.dxbc
; RUN: llvm-dis %t.dxil.bc -o - | FileCheck %S/SemanticSignatures-vs.ll --check-prefixes=MD,OPS
; RUN: llvm-dis %t.ildb.bc -o - | FileCheck %S/SemanticSignatures-vs.ll --check-prefixes=MD,OPS
; RUN: llvm-dis %t.ildb.bc -o - | FileCheck %s

; Both cloned bitcode modules and the container must retain the same finalized
; signature after source metadata stripping, op lowering, and debug preparation.
; CHECK: !llvm.dbg.cu =
; CHECK: !DICompileUnit

!llvm.dbg.cu = !{!200}
!llvm.module.flags = !{!202, !203}
!200 = distinct !DICompileUnit(language: DW_LANG_C99, file: !201, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!201 = !DIFile(filename: "signature.hlsl", directory: "/")
!202 = !{i32 2, !"Dwarf Version", i32 4}
!203 = !{i32 2, !"Debug Info Version", i32 3}

; RUN: opt %s -dxil-pdb -o /dev/null
; RUN: llvm-pdbutil pdb2yaml --dxcontainer PDBPartsTest-dxil.pdb | FileCheck %s

; Check that PDB file contains only debug-info relevant parts.
; CHECK:       PartCount:       2
; CHECK:     Parts:
; CHECK:       - Name:            DXIL
; CHECK:       - Name:            ILDN

target triple = "dxilv1.3-pc-shadermodel6.3-library"

@dx.dxil = private constant [4 x i8] c"BC\C0\DE", section "DXIL", align 4
@dx.ildn = private constant [26 x i8] c"\00\00\15\00PDBPartsTest-dxil.pdb\00", section "ILDN", align 4
@dx.pdb.name = private constant [21 x i8] c"PDBPartsTest-dxil.pdb", section "PDBNAME", align 4
@dx.pdb.hash = private constant [16 x i8] c"?\B9Z\96(\94D*{\AA&\A0P\B3\C9\D7", section "PDBHASH", align 4
@llvm.compiler.used = appending global [4 x ptr] [ptr @dx.dxil, ptr @dx.ildn, ptr @dx.pdb.name, ptr @dx.pdb.hash], section "llvm.metadata"

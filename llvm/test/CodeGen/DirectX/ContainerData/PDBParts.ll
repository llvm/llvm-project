; RUN: opt %s -dxil-pdb -o /dev/null
; RUN: llvm-pdbutil pdb2yaml --dxcontainer PDBPartsTest.pdb | FileCheck %s

; Check that PDB file contains only debug-info relevant parts.
; CHECK:       PartCount:       5
; CHECK:     Parts:
; CHECK-DAG:       - Name:            ILDB
; CHECK-DAG:       - Name:            ILDN
; CHECK-DAG:       - Name:            HASH
; CHECK-DAG:       - Name:            SRCI
; CHECK-DAG:       - Name:            VERS

target triple = "dxilv1.3-pc-shadermodel6.3-library"

@dx.dxil = private constant [4 x i8] c"BC\C0\DE", section "DXIL", align 4
@dx.ildb = private constant [4 x i8] c"BC\C0\DE", section "ILDB", align 4
@dx.ildn = private constant [21 x i8] c"\00\00\10\00PDBPartsTest.pdb\00", section "ILDN", align 4
@dx.hash = private constant [20 x i8] c"\01\00\00\00?\B9Z\96(\94D*{\AA&\A0P\B3\C9\D7", section "HASH", align 4
@dx.srci = private constant [76 x i8] c"\4C\00\00\00\00\00\03\00\14\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\1C\00\00\00\00\00\00\00\1C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\14\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00", section "SRCI", align 4
@dx.vers = private constant [64 x i8] c"\16\00\01\00\01\00\00\00\EE\A0\08\000\00\00\0026814cdaaa210c3e8aeea9112be2768964e6bdec\0022.1.1\00", section "VERS", align 4
@dx.pdb.name = private constant [16 x i8] c"PDBPartsTest.pdb", section "PDBNAME", align 4
@dx.pdb.hash = private constant [16 x i8] c"?\B9Z\96(\94D*{\AA&\A0P\B3\C9\D7", section "PDBHASH", align 4
@dx.sfi0 = private constant i64 0, section "SFI0", align 4
@dx.isg1 = private constant [8 x i8] c"\00\00\00\00\08\00\00\00", section "ISG1", align 4
@dx.osg1 = private constant [8 x i8] c"\00\00\00\00\08\00\00\00", section "OSG1", align 4
@llvm.compiler.used = appending global [11 x ptr] [ptr @dx.dxil, ptr @dx.ildb, ptr @dx.ildn, ptr @dx.hash, ptr @dx.srci, ptr @dx.vers, ptr @dx.pdb.name, ptr @dx.pdb.hash, ptr @dx.sfi0, ptr @dx.isg1, ptr @dx.osg1], section "llvm.metadata"

; RUN: opt %s -dxil-pdb --dx-pdb-in-private -S -o - | FileCheck %s --check-prefix=CHECK-IR
; RUN: opt %s -dxil-pdb --dx-pdb-in-private -o /dev/null
; RUN: llvm-pdbutil pdb2yaml --dxcontainer PdbInPrivateTest.pdb | FileCheck %s --check-prefix=CHECK-PDB

;; Check that dxil-pdb pass emits a PRIV global when --dx-pdb-in-private is set.
; CHECK-IR: @dx.priv = private constant {{.*}} section "PRIV"

; Check that the companion PDB file is still written when PDBNAME is present.
; CHECK-PDB:   PartCount: 1
; CHECK-PDB: Parts:
; CHECK-PDB:   - Name:    ILDB

target triple = "dxilv1.3-pc-shadermodel6.3-library"

@dx.ildb = private constant [4 x i8] c"BC\C0\DE", section "ILDB", align 4
@dx.pdb.name = private constant [20 x i8] c"PdbInPrivateTest.pdb", section "PDBNAME", align 4
@dx.pdb.hash = private constant [16 x i8] c"dummymodulehash!", section "PDBHASH", align 4
@llvm.compiler.used = appending global [3 x ptr] [ptr @dx.ildb, ptr @dx.pdb.name, ptr @dx.pdb.hash], section "llvm.metadata"

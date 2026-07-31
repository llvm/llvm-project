;; Check that dxil-pdb can emit PRIV using a temporary PDB when PDBNAME is absent.
; RUN: opt %s -dxil-pdb --dx-pdb-in-private -S -o - | FileCheck %s

;; CHECK: @dx.priv = private constant {{.*}} section "PRIV"

target triple = "dxilv1.3-pc-shadermodel6.3-library"

@dx.ildb = private constant [4 x i8] c"BC\C0\DE", section "ILDB", align 4
@dx.pdb.hash = private constant [16 x i8] c"dummymodulehash!", section "PDBHASH", align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr @dx.ildb, ptr @dx.pdb.hash], section "llvm.metadata"

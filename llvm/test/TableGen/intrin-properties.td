// RUN: sed -e 's/<PROP>/ArgMemOnly/;s/<ATTR>/ArgMemOnly/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/Cold/;s/<ATTR>/Cold/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/Convergent/;s/<ATTR>/Convergent/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/InaccessibleMemOnly/;s/<ATTR>/InaccessibleMemOnly/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/InaccessibleMemOrArgMemOnly/;s/<ATTR>/InaccessibleMemOrArgMemOnly/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoDuplicate/;s/<ATTR>/NoDuplicate/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoFree/;s/<ATTR>/NoFree/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoMem/;s/<ATTR>/ReadNone/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoMerge/;s/<ATTR>/NoMerge/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoReturn/;s/<ATTR>/NoReturn/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/NoSync/;s/<ATTR>/NoSync/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/ReadMem/;s/<ATTR>/ReadOnly/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/Speculatable/;s/<ATTR>/Speculatable/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/WillReturn/;s/<ATTR>/WillReturn/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t
// RUN: sed -e 's/<PROP>/WriteMem/;s/<ATTR>/WriteOnly/' < %s > %t; llvm-tblgen -gen-intrinsic-impl -I %p/../../include %t | FileCheck %t

include "llvm/IR/Intrinsics.td"

// CHECK: [[I:[0-9]+]], // llvm.tgtest.Intr<PROP>
// CHECK: case [[I]]:
// CHECK-NEXT: Atts[] = {{.*}}Attribute::<ATTR>
def int_tgtest_Intr<PROP>:
  Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [Intr<PROP>]>;


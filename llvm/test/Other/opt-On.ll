; RUN: not opt -O1 -O2 < %s 2>&1 | FileCheck %s --check-prefix=MULTIPLE
; RUN: not opt -O1 -passes='no-op-module' < %s 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: not opt -O1 --gvn < %s 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt -O0 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -O1 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -O2 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -O3 < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -Os < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -Oz < %s -S 2>&1 | FileCheck %s --check-prefix=OPT
; RUN: opt -O2 -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AA

; MULTIPLE: Cannot specify multiple -O#
; BOTH: Cannot specify -O# and --passes=
; OPT: define void @f
; Make sure we run the default AA pipeline with `opt -O#`
; AA: Running analysis: ScopedNoAliasAA

define void @f() {
  unreachable
}

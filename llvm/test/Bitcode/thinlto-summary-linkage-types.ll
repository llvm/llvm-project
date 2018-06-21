; Check the linkage types in both the per-module and combined summaries.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED

define private void @private()
; CHECK: <PERMODULE {{.*}} op1=8
; COMBINED-DAG: <COMBINED {{.*}} op2=8
{
  ret void
}

define internal void @internal()
; CHECK: <PERMODULE {{.*}} op1=7
; COMBINED-DAG: <COMBINED {{.*}} op2=7
{
  ret void
}

define available_externally void @available_externally()
; CHECK: <PERMODULE {{.*}} op1=1
; COMBINED-DAG: <COMBINED {{.*}} op2=1
{
  ret void
}

define linkonce void @linkonce()
; CHECK: <PERMODULE {{.*}} op1=2
; COMBINED-DAG: <COMBINED {{.*}} op2=2
{
  ret void
}

define weak void @weak()
; CHECK: <PERMODULE {{.*}} op1=4
; COMBINED-DAG: <COMBINED {{.*}} op2=4
{
  ret void
}

define linkonce_odr void @linkonce_odr()
; CHECK: <PERMODULE {{.*}} op1=3
; COMBINED-DAG: <COMBINED {{.*}} op2=3
{
  ret void
}

define weak_odr void @weak_odr()
; CHECK: <PERMODULE {{.*}} op1=5
; COMBINED-DAG: <COMBINED {{.*}} op2=5
{
  ret void
}

define external void @external()
; CHECK: <PERMODULE {{.*}} op1=0
; COMBINED-DAG: <COMBINED {{.*}} op2=0
{
  ret void
}

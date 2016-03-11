; Check the linkage types in both the per-module and combined summaries.
; RUN: llvm-as -function-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED

define private void @private()
; CHECK: <PERMODULE {{.*}} op1=9
; COMBINED-DAG: <COMBINED {{.*}} op1=9
{
  ret void
}

define internal void @internal()
; CHECK: <PERMODULE {{.*}} op1=3
; COMBINED-DAG: <COMBINED {{.*}} op1=3
{
  ret void
}

define available_externally void @available_externally()
; CHECK: <PERMODULE {{.*}} op1=12
; COMBINED-DAG: <COMBINED {{.*}} op1=12
{
  ret void
}

define linkonce void @linkonce()
; CHECK: <PERMODULE {{.*}} op1=18
; COMBINED-DAG: <COMBINED {{.*}} op1=18
{
  ret void
}

define weak void @weak()
; CHECK: <PERMODULE {{.*}} op1=16
; COMBINED-DAG: <COMBINED {{.*}} op1=16
{
  ret void
}

define linkonce_odr void @linkonce_odr()
; CHECK: <PERMODULE {{.*}} op1=19
; COMBINED-DAG: <COMBINED {{.*}} op1=19
{
  ret void
}

define weak_odr void @weak_odr()
; CHECK: <PERMODULE {{.*}} op1=17
; COMBINED-DAG: <COMBINED {{.*}} op1=17
{
  ret void
}

define external void @external()
; CHECK: <PERMODULE {{.*}} op1=0
; COMBINED-DAG: <COMBINED {{.*}} op1=0
{
  ret void
}

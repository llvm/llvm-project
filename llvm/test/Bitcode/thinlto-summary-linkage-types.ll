; Check the linkage types in both the per-module and combined summaries.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED

define private void @private()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2120
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2120
{
  ret void
}

define internal void @internal()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2119
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2119
{
  ret void
}

define available_externally void @available_externally()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2049
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2049
{
  ret void
}

define linkonce void @linkonce()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2050
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2050
{
  ret void
}

define weak void @weak()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2052
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2052
{
  ret void
}

define linkonce_odr void @linkonce_odr()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2051
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2051
{
  ret void
}

define weak_odr void @weak_odr()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2053
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2053
{
  ret void
}

define external void @external()
; CHECK: <PERMODULE_PROFILE {{.*}} op1=2048
; COMBINED-DAG: <COMBINED_PROFILE {{.*}} op2=2048
{
  ret void
}

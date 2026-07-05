; RUN: llc -O0 -mtriple=hexagon-- -relocation-model=static < %s | FileCheck %s

@dst_le = dso_local thread_local global i32 0, align 4
@src_le = dso_local thread_local global i32 0, align 4

; CHECK-LABEL: test_local_exec
; CHECK-DAG:   ##src_le@TPREL
; CHECK-DAG:   ##dst_le@TPREL
define dso_local i32 @test_local_exec() nounwind {
entry:
  %0 = load i32, ptr @src_le, align 4
  store i32 %0, ptr @dst_le, align 4
  ret i32 0
}

@dst_ie = external thread_local global i32
@src_ie = external thread_local global i32

; CHECK-LABEL: test_initial_exec:
; CHECK-DAG:   = memw(##src_ie@IE)
; CHECK-DAG:   = memw(##dst_ie@IE)
define dso_local i32 @test_initial_exec() nounwind {
entry:
  %0 = load i32, ptr @src_ie, align 4
  store i32 %0, ptr @dst_ie, align 4
  ret i32 0
}


; RUN: llc -O0 -march=hexagon -relocation-model=pic < %s | FileCheck %s

@dst_ie = thread_local(initialexec) global i32 0, align 4
@src_ie = thread_local(initialexec) global i32 0, align 4

; CHECK-LABEL:    test_initial_exec
; CHECK-DAG:      = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK-DAG:      ##src_ie@IEGOT
; CHECK-DAG:      ##dst_ie@IEGOT
; CHECK-NOT:  call
define i32 @test_initial_exec() nounwind {
entry:
  %0 = load i32, ptr @src_ie, align 4
  store i32 %0, ptr @dst_ie, align 4
  ret i32 0
}

@dst_gd = external thread_local global i32
@src_gd = external thread_local global i32

; At the moment, the local-dynamic model uses the same code as the
; general-dynamic model.

; CHECK-LABEL: test_dynamic
; CHECK-DAG:   = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK-DAG:   ##src_gd@GDGOT
; CHECK-DAG:   ##dst_gd@GDGOT
; CHECK-DAG:   call src_gd@GDPLT
; CHECK-DAG:   call dst_gd@GDPLT

define i32 @test_dynamic() nounwind {
entry:
  %0 = load i32, ptr @src_gd, align 4
  store i32 %0, ptr @dst_gd, align 4
  ret i32 0
}


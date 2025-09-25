; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | \
; RUN: FileCheck %s --check-prefix=32BIT
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 -mattr=-altivec | \
; RUN: FileCheck %s --check-prefix=64BIT

define void @create_trampoline(ptr %buffer, ptr %nval) nounwind {
entry:
  call void @llvm.init.trampoline(ptr %buffer, ptr @nested , ptr %nval)
  ret void
}

declare i32 @nested(i32);

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind

; 32BIT:     stw 4, 8(3)
; 32BIT:     lwz [[FuncDesc:[0-9]+]], L..C0(2)
; 32BIT-DAG: lwz [[SCRATCH1:[0-9]+]], 0([[FuncDesc]])
; 32BIT-DAG: lwz [[SCRATCH2:[0-9]+]], 4([[FuncDesc]])
; 32BIT-DAG: stw [[SCRATCH1]], 0(3)
; 32BIT-DAG: stw [[SCRATCH2]], 4(3)

; 64BIT:     std 4, 16(3)
; 64BIT-DAG: ld [[FuncDesc:[0-9]+]], L..C0(2)
; 64BIT-DAG: ld [[SCRATCH1:[0-9]+]], 0([[FuncDesc]])
; 64BIT-DAG: ld [[SCRATCH2:[0-9]+]], 8([[FuncDesc]])
; 64BIT-DAG: std [[SCRATCH1]], 0(3)
; 64BIT-DAG: std [[SCRATCH2]], 8(3)

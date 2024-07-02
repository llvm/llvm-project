; Test upgrade of dbg.addr intrinsics into dbg.value with DW_OP_deref appended
;
; RUN: llvm-dis < %s.bc --write-experimental-debuginfo=false | FileCheck %s
; RUN: llvm-dis < %s.bc --load-bitcode-into-experimental-debuginfo-iterators --write-experimental-debuginfo=false | FileCheck %s
; RUN: verify-uselistorder < %s.bc

define i32 @example(i32 %num) {
entry:
  %num.addr = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
  ; CHECK-NOT: call void @llvm.dbg.addr
  ; CHECK: call void @llvm.dbg.value(metadata ptr %num.addr, metadata ![[#]], metadata !DIExpression(DW_OP_deref))
  call void @llvm.dbg.addr(metadata ptr %num.addr, metadata !16, metadata !DIExpression(DW_OP_plus_uconst, 0)), !dbg !17
  %0 = load i32, ptr %num.addr, align 4
  ret i32 %0
}

; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.addr(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "/app/example.c", directory: "/app")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "example", scope: !11, file: !11, line: 1, type: !12, scopeLine: 1, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "example.c", directory: "/app")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocalVariable(name: "num", arg: 1, scope: !10, file: !11, line: 1, type: !14)
!17 = !DILocation(line: 1, column: 17, scope: !10)

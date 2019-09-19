; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

; CHECK: DIBlockByRefStruct on DICompositeType is no longer supported
; CHECK: warning: ignoring invalid debug info

define void @foo() {
entry:
  %s = alloca i32
  call void @llvm.dbg.declare(metadata i32* %s, metadata !2, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram()
!2 = !DILocalVariable(scope: !1, type: !3)
!3 = !DICompositeType(tag: DW_TAG_structure_type, flags: DIFlagReservedBit4)

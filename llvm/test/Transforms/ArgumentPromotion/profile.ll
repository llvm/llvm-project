; RUN: opt -argpromotion -mem2reg -S < %s | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Checks if !prof metadata is correct after argpromotion

define void @caller() #0 {
  %x = alloca i32
  store i32 42, i32* %x
  call void @promote_i32_ptr(i32* %x), !prof !6
; CHECK: call void @promote_i32_ptr(i32 42), !prof ![[PROF:[0-9]+]]
  ret void
}

; CHECK: define internal void @promote_i32_ptr(i32 %xp.val) !dbg ![[DBG:[0-9]+]] !prof ![[FUNCPROF:[0-9]+]]
define internal void @promote_i32_ptr(i32* %xp) !dbg !7 !prof !12 {
  %x = load i32, i32* %xp
  call void @use_i32(i32 %x)
  ret void
}

declare void @use_i32(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (trunk 353186) (llvm/trunk 353190)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "profile.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
; CHECK: ![[PROF]] = !{!"branch_weights", i32 30}
!6 = !{!"branch_weights", i32 30}
; CHECK: ![[DBG]] = distinct !DISubprogram(name: "promote_i32_ptr"
!7 = distinct !DISubprogram(name: "promote_i32_ptr", linkageName: "_ZL15promote_i32_ptrPi", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK: ![[FUNCPROF]] = !{!"function_entry_count", i64 331}
!12 = !{!"function_entry_count", i64 331}

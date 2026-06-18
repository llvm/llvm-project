; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; Simulates a nocall function after DeadArgElimination removed 'ctx':
;   static __noinline void func(void *ctx, int key, int sig) { ... }
; 'ctx' is dead.  'key' is stack-homed (address passed to helper),
; described via #dbg_assign.  'sig' is register-only, described via
; #dbg_value.  BTF FUNC_PROTO should list both key and sig.

; CHECK:      [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [3] FUNC_PROTO '(anon)' ret_type_id=0 vlen=2
; CHECK-NEXT: 	'key' type_id=2
; CHECK-NEXT: 	'sig' type_id=2
; CHECK-NEXT: [4] FUNC 'func' type_id=3 linkage=static

@map = external global i64

define internal void @func(i32 noundef %key, i32 noundef %sig) #0 !dbg !7 {
entry:
  %key.addr = alloca i32, align 4, !DIAssignID !20
    #dbg_assign(i1 poison, !13, !DIExpression(), !20, ptr %key.addr, !DIExpression(), !19)
  store i32 %key, ptr %key.addr, align 4, !DIAssignID !21
    #dbg_assign(i32 %key, !13, !DIExpression(), !21, ptr %key.addr, !DIExpression(), !19)
    #dbg_value(i32 %sig, !14, !DIExpression(), !19)
  %call = call ptr inttoptr (i64 1 to ptr)(ptr noundef @map, ptr noundef %key.addr) #1
  ret void, !dbg !22
}

attributes #0 = { noinline nounwind }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !23}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/DNE")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(cc: DW_CC_nocall, types: !9)
!9 = !{null, !15, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "ctx", arg: 1, scope: !7, file: !1, line: 1, type: !15)
!13 = !DILocalVariable(name: "key", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "sig", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!19 = !DILocation(line: 1, column: 1, scope: !7)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()
!22 = !DILocation(line: 2, column: 1, scope: !7)
!23 = !{i32 7, !"debug-info-assignment-tracking", i1 true}

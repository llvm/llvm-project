;; Test we merge non-inlined profile only once with '-sample-profile-merge-inlinee'
; RUN: opt < %s -passes='function(callsite-splitting),sample-profile' -sample-profile-file=%S/Inputs/inline-mergeprof.prof -sample-profile-merge-inlinee=true -S | FileCheck %s

%struct.bitmap = type { i32, %struct.bitmap* }

; CHECK-LABEL: @main
define void @main(i1 %c, %struct.bitmap* %a_elt, %struct.bitmap* %b_elt) #0 !dbg !6 {
entry:
  br label %Top

Top:
  %tobool1 = icmp eq %struct.bitmap* %a_elt, null
  br i1 %tobool1, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne %struct.bitmap* %b_elt, null
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  %p = phi i1 [0, %Top], [%c, %NextCond]
;; The call site is replicated by callsite-splitting pass and they end up share the same sample profile
; CHECK: call void @_Z3sumii(%struct.bitmap* null, %struct.bitmap* null, %struct.bitmap* %b_elt, i1 false)
; CHECK: call void @_Z3sumii(%struct.bitmap* nonnull %a_elt, %struct.bitmap* nonnull %a_elt, %struct.bitmap* nonnull %b_elt, i1 %c)
  call void @_Z3sumii(%struct.bitmap* %a_elt, %struct.bitmap* %a_elt, %struct.bitmap* %b_elt, i1 %p), !dbg !8
  br label %End

End:
  ret void
}

define void @_Z3sumii(%struct.bitmap* %dst_elt, %struct.bitmap* %a_elt, %struct.bitmap* %b_elt, i1 %c)  #0 !dbg !12 {
entry:
  %tobool = icmp ne %struct.bitmap* %a_elt, null
  %tobool1 = icmp ne %struct.bitmap* %b_elt, null
  %or.cond = and i1 %tobool, %tobool1, !dbg !13
  br i1 %or.cond, label %Cond, label %Big

Cond:
  %cmp = icmp eq %struct.bitmap*  %dst_elt, %a_elt, !dbg !14
  br i1 %cmp, label %Small, label %Big, !dbg !15

Small:
  br label %End

Big:
  br label %End

End:
  ret void
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.5 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !7, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 10, scope: !9)
!9 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 2)
!10 = distinct !DILexicalBlock(scope: !6, file: !1, line: 10)
!11 = !DILocation(line: 12, scope: !6)
!12 = distinct !DISubprogram(name: "sum", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 4, scope: !12)
!14 = !DILocation(line: 5, scope: !12)
!15 = !DILocation(line: 6, scope: !12)


;; Check the profile of funciton sum is only merged once though the original callsite is replicted.
; CHECK: name: "sum"
; CHECK-NEXT: {!"function_entry_count", i64 46}
; CHECK: !{!"branch_weights", i32 11, i32 37}
; CHECK: !{!"branch_weights", i32 11, i32 1}

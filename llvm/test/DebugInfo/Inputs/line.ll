; From source:
; int f(int a, int b) {
;   return a   //
;          &&  //
;          b;
; }

; Check that the comparison of 'a' is attributed to line 2, not 3.

; CHECK: .loc{{ +}}1{{ +}}2
; CHECK-NOT: .loc{{ }}
; CHECK: cmp

; Function Attrs: nounwind uwtable
define i32 @_Z1fii(i32 %a, i32 %b) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4, !dbg !10
  %tobool = icmp ne i32 %0, 0, !dbg !10
  br i1 %tobool, label %land.rhs, label %land.end, !dbg !11

land.rhs:                                         ; preds = %entry
  %1 = load i32, ptr %b.addr, align 4, !dbg !12
  %tobool1 = icmp ne i32 %1, 0, !dbg !12
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %tobool1, %land.rhs ]
  %conv = zext i1 %2 to i32, !dbg !10
  ret i32 %conv, !dbg !13
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 (trunk 227472) (llvm/trunk 227476)", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "line.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "line.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.7.0 (trunk 227472) (llvm/trunk 227476)"}
!10 = !DILocation(line: 2, scope: !4)
!11 = !DILocation(line: 3, scope: !4)
!12 = !DILocation(line: 4, scope: !4)
!13 = !DILocation(line: 2, scope: !4)

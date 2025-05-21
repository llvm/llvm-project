; RUN: opt %s -passes=inline -S -o - | FileCheck %s

;; Inline `f` into `g`. The inlined assignment store and add should retain
;; their atom info.

; CHECK: _Z1gi
; CHECK-NOT: _Z1fi
; CHECK: %add.i = add nsw i32 %mul.i, 1, !dbg [[G1R2:!.*]]
; CHECK-NEXT: store i32 %add.i, ptr %x.i, align 4, !dbg [[G1R1:!.*]]

; CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
; CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)

define hidden void @_Z1fi(i32 noundef %a) !dbg !11 {
entry:
  %a.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %mul = mul nsw i32 %0, 2
  %add = add nsw i32 %mul, 1, !dbg !19
  store i32 %add, ptr %x, align 4, !dbg !20
  ret void
}

define hidden void @_Z1gi(i32 noundef %b) !dbg !23 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %b.addr, align 4
  call void @_Z1fi(i32 noundef %0), !dbg !24
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 19.0.0"}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!19 = !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 2)
!20 = !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 1)
!23 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 5, scope: !23)


; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s

; CHECK-LABEL: _Z3foov:
; CHECK: .loc    1 4 3 prologue_end
; CHECK: .cfi_escape 0x2e, 0x10
define void @_Z3foov() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !4 {
entry:
  tail call void @_Z3bariii(i32 0, i32 1, i32 2) #1, !dbg !10
  invoke void @_Z3bariii(i32 4, i32 5, i32 6) #1
          to label %try.cont unwind label %lpad, !dbg !11

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null, !dbg !13
  %1 = extractvalue { i8*, i32 } %0, 0, !dbg !13
  %2 = tail call i8* @__cxa_begin_catch(i8* %1) #2, !dbg !14
  tail call void @__cxa_end_catch(), !dbg !15
  br label %try.cont, !dbg !15

try.cont:                                         ; preds = %entry, %lpad
  ret void, !dbg !17
}

; Function Attrs: optsize
declare void @_Z3bariii(i32, i32, i32) #0

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #0 = { optsize "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { optsize }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 249520)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "foo")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 249520)"}
!10 = !DILocation(line: 4, column: 3, scope: !4)
!11 = !DILocation(line: 6, column: 5, scope: !12)
!12 = distinct !DILexicalBlock(scope: !4, file: !1, line: 5, column: 7)
!13 = !DILocation(line: 10, column: 1, scope: !12)
!14 = !DILocation(line: 7, column: 3, scope: !12)
!15 = !DILocation(line: 9, column: 3, scope: !16)
!16 = distinct !DILexicalBlock(scope: !4, file: !1, line: 7, column: 17)
!17 = !DILocation(line: 10, column: 1, scope: !4)

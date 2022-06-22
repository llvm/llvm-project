; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s
;
; Generated at -O3 from:
; g();f(){for(;;){g();}}g(){__builtin___memset_chk(0,0,0,__builtin_object_size(1,0));}
; The fact that everything is on one line is significant!
;
; This test ensures that inline info isn't dropped even if the call site and the
; inlined function are defined on the same line.
source_filename = "t.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

; Function Attrs: noreturn nounwind ssp
define i32 @f() local_unnamed_addr #0 !dbg !7 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64.p0(ptr inttoptr (i64 1 to ptr), i1 false) #2, !dbg !11
  br label %for.cond, !dbg !18

for.cond:                                         ; preds = %for.cond, %entry
  ; CHECK: %call.i
  %call.i = tail call ptr @__memset_chk(ptr null, i32 0, i64 0, i64 %0) #2, !dbg !19 
  ; CHECK: br label %for.cond, !dbg ![[BR:[0-9]+]]
  br label %for.cond, !dbg !20, !llvm.loop !21
}

; Function Attrs: nounwind ssp
define i32 @g() local_unnamed_addr #1 !dbg !12 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64.p0(ptr inttoptr (i64 1 to ptr), i1 false), !dbg !22
  %call = tail call ptr @__memset_chk(ptr null, i32 0, i64 0, i64 %0) #2, !dbg !23
  ret i32 undef, !dbg !24
}

; Function Attrs: nounwind
declare ptr @__memset_chk(ptr, i32, i64, i64) local_unnamed_addr #2

; Function Attrs: nounwind readnone
declare i64 @llvm.objectsize.i64.p0(ptr, i1) #3

attributes #0 = { noreturn nounwind ssp }
attributes #1 = { nounwind ssp  }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM version 4.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"LLVM version 4.0.0"}
; CHECK: ![[F:.*]] = distinct !DISubprogram(name: "f",
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 56, scope: !12, inlinedAt: !13)
!12 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!13 = distinct !DILocation(line: 1, column: 17, scope: !14)
; CHECK: ![[BF:.*]] = !DILexicalBlockFile(scope: ![[LB1:[0-9]+]],
; CHECK-SAME:                             discriminator: 2)
!14 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 2)
; CHECK: ![[LB1]] = distinct !DILexicalBlock(scope: ![[LB2:[0-9]+]],
; CHECK-SAME:                                line: 1, column: 16)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 1, column: 16)
; CHECK: ![[LB2]] = distinct !DILexicalBlock(scope: ![[LB3:[0-9]+]],
; CHECK-SAME:                                line: 1, column: 9)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 1, column: 9)
; CHECK: ![[LB3]] = distinct !DILexicalBlock(scope: ![[F]],
; CHECK-SAME:                                line: 1, column: 9)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 1, column: 9)
!18 = !DILocation(line: 1, column: 9, scope: !7)
!19 = !DILocation(line: 1, column: 27, scope: !12, inlinedAt: !13)
; CHECK: ![[BR]] =  !DILocation(line: 1, column: 9, scope: !14)
!20 = !DILocation(line: 1, column: 9, scope: !14)
!21 = distinct !{!21, !18}
!22 = !DILocation(line: 1, column: 56, scope: !12)
!23 = !DILocation(line: 1, column: 27, scope: !12)
!24 = !DILocation(line: 1, column: 84, scope: !12)

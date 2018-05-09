; RUN: llc < %s | FileCheck %s

; Verify that we get *two* .debug$S sections, the main one describing bar and
; main, and one for f and fin$f, which is comdat with f.

; Start in the main symbol section describing bar and main.

; CHECK: .section .debug$S,"dr"{{$}}
; CHECK: .long 4 # Debug section magic
; CHECK: # Symbol subsection for bar
; CHECK-NOT: Debug section magic
; CHECK: # Symbol subsection for main

; Emit symbol info for f and its associated code in a separate associated
; section.

; CHECK: .section .debug$S,"dr",associative,f{{$}}
; CHECK: .long 4 # Debug section magic
; CHECK: # Symbol subsection for f
; CHECK-NOT: Debug section magic
; CHECK: # Symbol subsection for ?fin$0@0@f@@

; Switch back to the main section for the shared file checksum table and string
; table.

; CHECK: .section .debug$S,"dr"{{$}}
; CHECK-NOT: Debug section magic
; CHECK: .cv_filechecksums
; CHECK: .cv_stringtable
; CHECK-NOT: .section .debug$S,

; Generated with this C++ source:
; void foo();
; void bar();
; extern volatile int x;
; inline void __declspec(noinline) f(bool c) {
;   x++;
;   if (c) {
;     __try {
;       foo();
;     } __finally {
;       x++;
;     }
;   } else
;     bar();
;   x++;
; }
; void bar() {
;   x++;
; }
; int main() {
;   f(true);
; }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

$f = comdat any

@x = external global i32, align 4

; Function Attrs: norecurse nounwind uwtable
define void @bar() #0 !dbg !7 {
entry:
  %0 = load volatile i32, i32* @x, align 4, !dbg !10, !tbaa !11
  %inc = add nsw i32 %0, 1, !dbg !10
  store volatile i32 %inc, i32* @x, align 4, !dbg !10, !tbaa !11
  ret void, !dbg !15
}

; Function Attrs: nounwind uwtable
define i32 @main() #1 !dbg !16 {
entry:
  tail call void @f(i32 1), !dbg !20
  ret i32 0, !dbg !21
}

; Function Attrs: inlinehint noinline nounwind uwtable
define linkonce_odr void @f(i32 %c) #2 comdat personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) !dbg !22 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %c, metadata !26, metadata !27), !dbg !28
  %0 = load volatile i32, i32* @x, align 4, !dbg !29, !tbaa !11
  %inc = add nsw i32 %0, 1, !dbg !29
  store volatile i32 %inc, i32* @x, align 4, !dbg !29, !tbaa !11
  %tobool = icmp eq i32 %c, 0, !dbg !30
  br i1 %tobool, label %if.else, label %if.then, !dbg !32

if.then:                                          ; preds = %entry
  invoke void bitcast (void (...)* @foo to void ()*)() #6
          to label %invoke.cont unwind label %ehcleanup, !dbg !33

invoke.cont:                                      ; preds = %if.then
  tail call fastcc void @"\01?fin$0@0@f@@"() #7, !dbg !36
  br label %if.end, !dbg !37

ehcleanup:                                        ; preds = %if.then
  %1 = cleanuppad within none [], !dbg !36
  tail call fastcc void @"\01?fin$0@0@f@@"() #7 [ "funclet"(token %1) ], !dbg !36
  cleanupret from %1 unwind to caller, !dbg !36

if.else:                                          ; preds = %entry
  tail call void @bar(), !dbg !38
  br label %if.end

if.end:                                           ; preds = %if.else, %invoke.cont
  %2 = load volatile i32, i32* @x, align 4, !dbg !39, !tbaa !11
  %inc1 = add nsw i32 %2, 1, !dbg !39
  store volatile i32 %inc1, i32* @x, align 4, !dbg !39, !tbaa !11
  ret void, !dbg !40
}

; Function Attrs: nounwind
define internal fastcc void @"\01?fin$0@0@f@@"() unnamed_addr #3 comdat($f) !dbg !41 {
entry:
  tail call void @llvm.dbg.value(metadata i8* null, metadata !44, metadata !27), !dbg !48
  tail call void @llvm.dbg.value(metadata i8 0, metadata !46, metadata !27), !dbg !48
  %0 = load volatile i32, i32* @x, align 4, !dbg !49, !tbaa !11
  %inc = add nsw i32 %0, 1, !dbg !49
  store volatile i32 %inc, i32* @x, align 4, !dbg !49, !tbaa !11
  ret void, !dbg !51
}

declare void @foo(...) #4

declare i32 @__C_specific_handler(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint noinline nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone }
attributes #6 = { noinline }
attributes #7 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 "}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 19, type: !8, isLocal: false, isDefinition: true, scopeLine: 19, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 20, column: 4, scope: !7)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 21, column: 1, scope: !7)
!16 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 22, type: !17, isLocal: false, isDefinition: true, scopeLine: 22, isOptimized: true, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{!19}
!19 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = !DILocation(line: 23, column: 3, scope: !16)
!21 = !DILocation(line: 24, column: 1, scope: !16)
!22 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 5, type: !23, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !19}
!25 = !{!26}
!26 = !DILocalVariable(name: "c", arg: 1, scope: !22, file: !1, line: 5, type: !19)
!27 = !DIExpression()
!28 = !DILocation(line: 5, column: 40, scope: !22)
!29 = !DILocation(line: 6, column: 4, scope: !22)
!30 = !DILocation(line: 7, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !22, file: !1, line: 7, column: 7)
!32 = !DILocation(line: 7, column: 7, scope: !22)
!33 = !DILocation(line: 9, column: 7, scope: !34)
!34 = distinct !DILexicalBlock(scope: !35, file: !1, line: 8, column: 11)
!35 = distinct !DILexicalBlock(scope: !31, file: !1, line: 7, column: 10)
!36 = !DILocation(line: 10, column: 5, scope: !34)
!37 = !DILocation(line: 13, column: 3, scope: !35)
!38 = !DILocation(line: 14, column: 5, scope: !31)
!39 = !DILocation(line: 15, column: 4, scope: !22)
!40 = !DILocation(line: 16, column: 1, scope: !22)
!41 = distinct !DISubprogram(linkageName: "\01?fin$0@0@f@@", scope: !1, file: !1, line: 10, type: !42, isLocal: true, isDefinition: true, scopeLine: 10, flags: DIFlagArtificial, isOptimized: true, unit: !0, retainedNodes: !43)
!42 = !DISubroutineType(types: !2)
!43 = !{!44, !46}
!44 = !DILocalVariable(name: "frame_pointer", arg: 2, scope: !41, type: !45, flags: DIFlagArtificial)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!46 = !DILocalVariable(name: "abnormal_termination", arg: 1, scope: !41, type: !47, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!48 = !DILocation(line: 0, scope: !41)
!49 = !DILocation(line: 11, column: 8, scope: !50)
!50 = distinct !DILexicalBlock(scope: !41, file: !1, line: 10, column: 17)
!51 = !DILocation(line: 12, column: 5, scope: !41)

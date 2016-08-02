; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test MachineScheduler handling of DBG_VALUE.
; rdar://12776937.
;
; CHECK: %if.else581
; CHECK: DEBUG_VALUE: num1
; CHECK: call

%union.rec = type {}

@.str15 = external hidden unnamed_addr constant [6 x i8], align 1

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @AttachGalley(%union.rec** nocapture %suspend_pt) nounwind uwtable ssp !dbg !21 {
entry:
  %num14075 = alloca [20 x i8], align 16
  br label %if.end33

if.end33:                                         ; preds = %entry
  %cmp1733 = icmp eq i32 undef, 0
  br label %if.else581

if.else581:                                       ; preds = %if.end33
  %cmp586 = icmp eq i8 undef, -123
  br i1 %cmp586, label %if.then588, label %if.else594

if.then588:                                       ; preds = %if.else581
  br label %for.cond1710.preheader

if.else594:                                       ; preds = %if.else581
  unreachable

for.cond1710.preheader:                           ; preds = %if.then588
  br label %for.cond1710

for.cond1710:                                     ; preds = %for.cond1710, %for.cond1710.preheader
  br i1 undef, label %for.cond1710, label %if.then3344

if.then3344:
  br label %if.then4073

if.then4073:                                      ; preds = %if.then3344
  call void @llvm.dbg.declare(metadata [20 x i8]* %num14075, metadata !4, metadata !DIExpression()), !dbg !DILocation(scope: !5)
  %arraydecay4078 = getelementptr inbounds [20 x i8], [20 x i8]* %num14075, i64 0, i64 0
  %0 = load i32, i32* undef, align 4
  %add4093 = add nsw i32 %0, 0
  %conv4094 = sitofp i32 %add4093 to float
  %div4095 = fdiv float %conv4094, 5.670000e+02
  %conv4096 = fpext float %div4095 to double
  %call4097 = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* %arraydecay4078, i32 0, i64 20, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str15, i64 0, i64 0), double %conv4096) nounwind
  br i1 %cmp1733, label %if.then4107, label %if.else4114

if.then4107:                                      ; preds = %if.then4073
  unreachable

if.else4114:                                      ; preds = %if.then4073
  unreachable
}

declare i32 @__sprintf_chk(i8*, i32, i64, i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 168918) (llvm/trunk 168920)", isOptimized: true, emissionKind: 0, file: !19, enums: !2, retainedTypes: !2, subprograms: !20, globals: !2)
!1 = !{!2}
!2 = !{}
!4 = !DILocalVariable(name: "num1", line: 815, scope: !5, file: !14, type: !15)
!5 = distinct !DILexicalBlock(line: 815, column: 0, file: !14, scope: !6)
!6 = distinct !DILexicalBlock(line: 812, column: 0, file: !14, scope: !7)
!7 = distinct !DILexicalBlock(line: 807, column: 0, file: !14, scope: !8)
!8 = distinct !DILexicalBlock(line: 440, column: 0, file: !14, scope: !9)
!9 = distinct !DILexicalBlock(line: 435, column: 0, file: !14, scope: !10)
!10 = distinct !DILexicalBlock(line: 434, column: 0, file: !14, scope: !11)
!11 = distinct !DILexicalBlock(line: 250, column: 0, file: !14, scope: !12)
!12 = distinct !DILexicalBlock(line: 249, column: 0, file: !14, scope: !13)
!13 = distinct !DILexicalBlock(line: 221, column: 0, file: !14, scope: !21)
!14 = !DIFile(filename: "MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c", directory: "MultiSource/Benchmarks/MiBench/consumer-typeset")
!15 = !DICompositeType(tag: DW_TAG_array_type, size: 160, align: 8, baseType: !16, elements: !17)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!17 = !{!18}
!18 = !DISubrange(count: 20)
!19 = !DIFile(filename: "MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c", directory: "MultiSource/Benchmarks/MiBench/consumer-typeset")

!20 = !{!21}
!21 = distinct !DISubprogram(name: "AttachGalley", isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1, file: !19, scope: !14, type: !22)
!22 = !DISubroutineType(types: !23)
!23 = !{null}

; Test DebugValue uses visited by RegisterPressureTracker findUseBetween().
;
; CHECK: @main
; CHECK: DEBUG_VALUE: main:X
; CHECK: call

%"class.__gnu_cxx::hash_map" = type { %"class.__gnu_cxx::hashtable" }
%"class.__gnu_cxx::hashtable" = type { i64, i64, i64, i64, i64, i64 }

define void @main() uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !37 {
entry:
  %X = alloca %"class.__gnu_cxx::hash_map", align 8
  br i1 undef, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  unreachable

cond.end:                                         ; preds = %entry
  call void @llvm.dbg.declare(metadata %"class.__gnu_cxx::hash_map"* %X, metadata !31, metadata !DIExpression()), !dbg !DILocation(scope: !37)
  %_M_num_elements.i.i.i.i = getelementptr inbounds %"class.__gnu_cxx::hash_map", %"class.__gnu_cxx::hash_map"* %X, i64 0, i32 0, i32 5
  invoke void @_Znwm()
          to label %exit.i unwind label %lpad2.i.i.i.i

exit.i:                                           ; preds = %cond.end
  unreachable

lpad2.i.i.i.i:                                    ; preds = %cond.end
  %0 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %lpad.body.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %lpad2.i.i.i.i
  unreachable

lpad.body.i.i:                                    ; preds = %lpad2.i.i.i.i
  resume { i8*, i32 } %0
}

declare i32 @__gxx_personality_v0(...)

declare void @_Znwm()

!llvm.dbg.cu = !{!30}

!30 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 169129) (llvm/trunk 169135)", isOptimized: true, emissionKind: 0, file: !34, enums: !2, retainedTypes: !2, subprograms: !36)
!31 = !DILocalVariable(name: "X", line: 29, scope: !37, type: !32)
!32 = !DIDerivedType(tag: DW_TAG_typedef, name: "HM", line: 28, file: !34, baseType: null)
!33 = !DIFile(filename: "SingleSource/Benchmarks/Shootout-C++/hash.cpp", directory: "SingleSource/Benchmarks/Shootout-C++")
!34 = !DIFile(filename: "SingleSource/Benchmarks/Shootout-C++/hash.cpp", directory: "SingleSource/Benchmarks/Shootout-C++")
!35 = !{i32 1, !"Debug Info Version", i32 3}
!36 = !{!37}
!37 = distinct !DISubprogram(name: "main", isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1, file: !19, scope: !14, type: !22)

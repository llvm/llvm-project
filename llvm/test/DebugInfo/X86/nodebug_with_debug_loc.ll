; REQUIRES: object-emission

; RUN: llc -mtriple=i386-linux-gnu -filetype=obj -relocation-model=pic %s -o /dev/null

; Derived from the test case in PR20367, there's nothing really positive to
; test here (hence no FileCheck, etc). All that was wrong is that the debug info
; intrinsics (introduced by inlining) in 'f1' were causing codegen to crash, but
; since 'f1' is a nodebug function, there's no positive outcome to confirm, just
; that debug info doesn't get in the way/cause a crash.

; The test case isn't particularly well reduced/tidy, but as simple as I could
; get the C++ source. I assume the complexity is mostly just about producing a
; certain amount of register pressure, so it might be able to be simplified/made
; more uniform.

; Generated from:
; $ clang-tot -cc1 -triple i386 -emit-obj -g -O3 repro.cpp
; void sink(const void *);
; int source();
; void f3(int);
; 
; extern bool b;
; 
; struct string {
;   unsigned *mem;
; };
; 
; extern string &str;
; 
; inline __attribute__((always_inline)) void s2(string *lhs) { sink(lhs->mem); }
; inline __attribute__((always_inline)) void f() {
;   string str2;
;   s2(&str2);
;   sink(&str2);
; }
; void __attribute__((nodebug)) f1() {
;   for (int iter = 0; iter != 2; ++iter) {
;     f();
;     sink(str.mem);
;     if (b) return;
;   }
; }

%struct.string = type { i32* }

@str = external constant %struct.string*
@b = external global i8

; Function Attrs: nounwind
define void @_Z2f1v() #0 {
entry:
  %str2.i = alloca %struct.string, align 4
  %0 = bitcast %struct.string* %str2.i to i8*, !dbg !26
  %1 = load %struct.string*, %struct.string** @str, align 4
  %mem = getelementptr inbounds %struct.string, %struct.string* %1, i32 0, i32 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iter.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @llvm.lifetime.start(i64 4, i8* %0), !dbg !26
  call void @llvm.dbg.value(metadata %struct.string* %str2.i, i64 0, metadata !16, metadata !DIExpression(DW_OP_deref)) #3, !dbg !26
  call void @llvm.dbg.value(metadata %struct.string* %str2.i, i64 0, metadata !27, metadata !DIExpression(DW_OP_deref)) #3, !dbg !29
  call void @_Z4sinkPKv(i8* undef) #3, !dbg !29
  call void @_Z4sinkPKv(i8* %0) #3, !dbg !30
  call void @llvm.lifetime.end(i64 4, i8* %0), !dbg !31
  %2 = load i32*, i32** %mem, align 4, !tbaa !32
  %3 = bitcast i32* %2 to i8*
  call void @_Z4sinkPKv(i8* %3) #3
  %4 = load i8, i8* @b, align 1, !tbaa !37, !range !39
  %tobool = icmp ne i8 %4, 0
  %inc = add nsw i32 %iter.02, 1
  %cmp = icmp eq i32 %inc, 2
  %or.cond = or i1 %tobool, %cmp
  br i1 %or.cond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @_Z4sinkPKv(i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "string", line: 7, size: 32, align: 32, file: !5, elements: !6, identifier: "_ZTS6string")
!5 = !DIFile(filename: "repro.cpp", directory: "/tmp/dbginfo")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "mem", line: 8, size: 32, align: 32, file: !5, scope: !4, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!11 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 14, file: !5, scope: !12, type: !13, variables: !15)
!12 = !DIFile(filename: "repro.cpp", directory: "/tmp/dbginfo")
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16}
!16 = !DILocalVariable(name: "str2", line: 15, scope: !11, file: !12, type: !4)
!17 = distinct !DISubprogram(name: "s2", linkageName: "_Z2s2P6string", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 13, file: !5, scope: !12, type: !18, variables: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: !4)
!21 = !{!22}
!22 = !DILocalVariable(name: "lhs", line: 13, arg: 1, scope: !17, file: !12, type: !20)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{!"clang version 3.5.0 "}
!26 = !DILocation(line: 15, scope: !11)
!27 = !DILocalVariable(name: "lhs", line: 13, arg: 1, scope: !17, file: !12, type: !20)
!28 = !DILocation(line: 16, scope: !11)
!29 = !DILocation(line: 13, scope: !17, inlinedAt: !28)
!30 = !DILocation(line: 17, scope: !11)
!31 = !DILocation(line: 18, scope: !11)
!32 = !{!33, !34, i64 0}
!33 = !{!"struct", !34, i64 0}
!34 = !{!"any pointer", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !{!38, !38, i64 0}
!38 = !{!"bool", !35, i64 0}
!39 = !{i8 0, i8 2}

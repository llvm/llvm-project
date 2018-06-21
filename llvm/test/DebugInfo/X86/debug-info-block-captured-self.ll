; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; Test that DW_AT_location is generated for a captured "self" inside a
; block.
;
; This test is split into two parts, the frontend part can be found at
; llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m
;
; CHECK: {{.*}}DW_AT_name{{.*}}_block_invoke{{.*}}
; CHECK: DW_TAG_variable
; CHECK-NOT:  DW_TAG
; CHECK:   DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name{{.*}}"self"{{.*}}
;
; CHECK: {{.*}}DW_AT_name{{.*}}_block_invoke{{.*}}
; CHECK: DW_TAG_variable
; CHECK-NOT:  DW_TAG
; CHECK:   DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name{{.*}}"self"{{.*}}
;
; Generated (and then reduced) from
; ----------------------------------------------------------------------
;
; @class T;
; @interface S
; @end
; @interface Mode
; -(int) count;
; @end
; @interface Context
; @end
; @interface ViewController
; @property (nonatomic, readwrite, strong) Context *context;
; @end
; typedef enum {
;     Unknown = 0,
; } State;
; @interface Main : ViewController
; {
;     T * t1;
;     T * t2;
; }
; @property(readwrite, nonatomic) State state;
; @end
; @implementation Main
; - (id) initWithContext:(Context *) context
; {
;     t1 = [self.context withBlock:^(id obj){
;         id *mode1;
; 	t2 = [mode1 withBlock:^(id object){
; 	    Mode *mode2 = object;
; 	    if ([mode2 count] != 0) {
; 	      self.state = 0;
; 	    }
; 	  }];
;       }];
; }
; @end
; ----------------------------------------------------------------------
; ModuleID = 'llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m'
%0 = type opaque
%struct.__block_descriptor = type { i64, i64 }
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
define internal void @"__24-[Main initWithContext:]_block_invoke"(i8* %.block_descriptor, i8* %obj) #0 !dbg !38 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !84
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !84
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, metadata !86, metadata !110), !dbg !87
  ret void, !dbg !87
}

define internal void @"__24-[Main initWithContext:]_block_invoke_2"(i8* %.block_descriptor, i8* %object) #0 !dbg !42 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !103
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !103
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, metadata !105, metadata !109), !dbg !106
  ret void, !dbg !106
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!108}
!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 3.3 ", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, file: !107, enums: !2, retainedTypes: !4, globals: !15, imports:  !15)
!1 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m", directory: "")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, line: 20, size: 32, align: 32, file: !107, elements: !4)
!4 = !{}
!15 = !{}
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "id", line: 31, file: !107, baseType: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !29)
!29 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_object", file: !107, elements: !30)
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "isa", size: 64, file: !107, scope: !29, baseType: !32)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !33)
!33 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_class", flags: DIFlagFwdDecl, file: !107)
!34 = !DICompositeType(tag: DW_TAG_structure_type, name: "Main", line: 23, flags: DIFlagArtificial | DIFlagObjectPointer, runtimeLang: DW_LANG_ObjC, file: !107)
!38 = distinct !DISubprogram(name: "__24-[Main initWithContext:]_block_invoke", line: 33, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 33, file: !1, scope: !1, type: !39, variables: !15)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !41, !27}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!42 = distinct !DISubprogram(name: "__24-[Main initWithContext:]_block_invoke_2", line: 35, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 35, file: !1, scope: !1, type: !39, variables: !15)
!84 = !DILocation(line: 33, scope: !38)
!86 = !DILocalVariable(name: "self", line: 41, scope: !38, file: !1, type: !34)
!87 = !DILocation(line: 41, scope: !38)
!103 = !DILocation(line: 35, scope: !42)
!105 = !DILocalVariable(name: "self", line: 40, scope: !42, file: !1, type: !34)
!106 = !DILocation(line: 40, scope: !42)
!107 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m", directory: "")
!108 = !{i32 1, !"Debug Info Version", i32 3}
!109 = !DIExpression(DW_OP_plus_uconst, 32, DW_OP_deref)
!110 = !DIExpression(DW_OP_plus_uconst, 32, DW_OP_deref)

; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; Test that DW_AT_location is generated for a captured "self" inside a
; block.
;
; This test is split into two parts, the frontend part can be found at
; llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m
;
; CHECK:      {{.*}}DW_AT_name{{.*}}_block_invoke{{.*}}
; CHECK:      DW_TAG_variable
; CHECK:      {{.*}}DW_AT_name{{.*}}"self"{{.*}}
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_location
;
; CHECK:      {{.*}}DW_AT_name{{.*}}_block_invoke{{.*}}
; CHECK:      DW_TAG_variable
; CHECK:      {{.*}}DW_AT_name{{.*}}"self"{{.*}}
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_location
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
declare void @llvm.dbg.declare(metadata, metadata) #1
define internal void @"__24-[Main initWithContext:]_block_invoke"(i8* %.block_descriptor, i8* %obj) #0 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !84
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !84
  call void @llvm.dbg.declare(metadata !{<{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block}, metadata !86), !dbg !87
  ret void, !dbg !87
}

define internal void @"__24-[Main initWithContext:]_block_invoke_2"(i8* %.block_descriptor, i8* %object) #0 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !103
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !103
  call void @llvm.dbg.declare(metadata !{<{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block}, metadata !105), !dbg !106
  ret void, !dbg !106
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!108}
!0 = metadata !{i32 786449, metadata !107, i32 16, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 2, metadata !2, metadata !4, metadata !23, metadata !15,  metadata !15, metadata !""} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m] [DW_LANG_ObjC]
!1 = metadata !{i32 786473, metadata !107} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786436, metadata !107, null, metadata !"", i32 20, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!4 = metadata !{}
!15 = metadata !{i32 0}
!23 = metadata !{metadata !38, metadata !42}
!27 = metadata !{i32 786454, metadata !107, null, metadata !"id", i32 31, i64 0, i64 0, i64 0, i32 0, metadata !28} ; [ DW_TAG_typedef ] [id] [line 31, size 0, align 0, offset 0] [from ]
!28 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !29} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_object]
!29 = metadata !{i32 786451, metadata !107, null, metadata !"objc_object", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_object] [line 0, size 0, align 0, offset 0] [def] [from ]
!30 = metadata !{metadata !31}
!31 = metadata !{i32 786445, metadata !107, metadata !29, metadata !"isa", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ] [isa] [line 0, size 64, align 0, offset 0] [from ]
!32 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !33} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from objc_class]
!33 = metadata !{i32 786451, metadata !107, null, metadata !"objc_class", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!34 = metadata !{i32 786451, metadata !107, null, metadata !"Main", i32 23, i64 0, i64 0, i32 0, i32 1092, null, i32 0, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [Main] [line 23, size 0, align 0, offset 0] [artificial] [decl] [from ]
!38 = metadata !{i32 786478, metadata !1, metadata !1, metadata !"__24-[Main initWithContext:]_block_invoke", metadata !"__24-[Main initWithContext:]_block_invoke", metadata !"", i32 33, metadata !39, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*, i8*)* @"__24-[Main initWithContext:]_block_invoke", null, null, metadata !15, i32 33} ; [ DW_TAG_subprogram ] [line 33] [local] [def] [__24-[Main initWithContext:]_block_invoke]
!39 = metadata !{i32 786453, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !40, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!40 = metadata !{null, metadata !41, metadata !27}
!41 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!42 = metadata !{i32 786478, metadata !1, metadata !1, metadata !"__24-[Main initWithContext:]_block_invoke_2", metadata !"__24-[Main initWithContext:]_block_invoke_2", metadata !"", i32 35, metadata !39, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*, i8*)* @"__24-[Main initWithContext:]_block_invoke_2", null, null, metadata !15, i32 35} ; [ DW_TAG_subprogram ] [line 35] [local] [def] [__24-[Main initWithContext:]_block_invoke_2]
!84 = metadata !{i32 33, i32 0, metadata !38, null}
!86 = metadata !{i32 786688, metadata !38, metadata !"self", metadata !1, i32 41, metadata !34, i32 0, i32 0, i64 1, i64 32} ; [ DW_TAG_auto_variable ] [self] [line 41]
!87 = metadata !{i32 41, i32 0, metadata !38, null}
!103 = metadata !{i32 35, i32 0, metadata !42, null}
!105 = metadata !{i32 786688, metadata !42, metadata !"self", metadata !1, i32 40, metadata !34, i32 0, i32 0, i64 1, i64 32} ; [ DW_TAG_auto_variable ] [self] [line 40]
!106 = metadata !{i32 40, i32 0, metadata !42, null}
!107 = metadata !{metadata !"llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m", metadata !""}
!108 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

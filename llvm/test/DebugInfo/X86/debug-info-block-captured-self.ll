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
define internal void @"__24-[Main initWithContext:]_block_invoke"(i8* %.block_descriptor, i8* %obj) #0 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !84
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !84
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, metadata !86, metadata !110), !dbg !87
  ret void, !dbg !87
}

define internal void @"__24-[Main initWithContext:]_block_invoke_2"(i8* %.block_descriptor, i8* %object) #0 {
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !103
  %block.captured-self = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, i32 0, i32 5, !dbg !103
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %block, metadata !105, metadata !109), !dbg !106
  ret void, !dbg !106
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!108}
!0 = !{!"0x11\0016\00clang version 3.3 \000\00\002\00\000", !107, !2, !4, !23, !15,  !15} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m] [DW_LANG_ObjC]
!1 = !{!"0x29", !107} ; [ DW_TAG_file_type ]
!2 = !{!3}
!3 = !{!"0x4\00\0020\0032\0032\000\000\000", !107, null, null, !4, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!4 = !{}
!15 = !{}
!23 = !{!38, !42}
!27 = !{!"0x16\00id\0031\000\000\000\000", !107, null, !28} ; [ DW_TAG_typedef ] [id] [line 31, size 0, align 0, offset 0] [from ]
!28 = !{!"0xf\00\000\0064\0064\000\000", null, null, !29} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_object]
!29 = !{!"0x13\00objc_object\000\000\000\000\000\000", !107, null, null, !30, null, null, null} ; [ DW_TAG_structure_type ] [objc_object] [line 0, size 0, align 0, offset 0] [def] [from ]
!30 = !{!31}
!31 = !{!"0xd\00isa\000\0064\000\000\000", !107, !29, !32} ; [ DW_TAG_member ] [isa] [line 0, size 64, align 0, offset 0] [from ]
!32 = !{!"0xf\00\000\0064\000\000\000", null, null, !33} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from objc_class]
!33 = !{!"0x13\00objc_class\000\000\000\000\004\000", !107, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!34 = !{!"0x13\00Main\0023\000\000\000\001092\0016", !107, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [Main] [line 23, size 0, align 0, offset 0] [artificial] [decl] [from ]
!38 = !{!"0x2e\00__24-[Main initWithContext:]_block_invoke\00__24-[Main initWithContext:]_block_invoke\00\0033\001\001\000\006\00256\000\0033", !1, !1, !39, null, void (i8*, i8*)* @"__24-[Main initWithContext:]_block_invoke", null, null, !15} ; [ DW_TAG_subprogram ] [line 33] [local] [def] [__24-[Main initWithContext:]_block_invoke]
!39 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !40, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!40 = !{null, !41, !27}
!41 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!42 = !{!"0x2e\00__24-[Main initWithContext:]_block_invoke_2\00__24-[Main initWithContext:]_block_invoke_2\00\0035\001\001\000\006\00256\000\0035", !1, !1, !39, null, void (i8*, i8*)* @"__24-[Main initWithContext:]_block_invoke_2", null, null, !15} ; [ DW_TAG_subprogram ] [line 35] [local] [def] [__24-[Main initWithContext:]_block_invoke_2]
!84 = !MDLocation(line: 33, scope: !38)
!86 = !{!"0x100\00self\0041\000", !38, !1, !34} ; [ DW_TAG_auto_variable ] [self] [line 41]
!87 = !MDLocation(line: 41, scope: !38)
!103 = !MDLocation(line: 35, scope: !42)
!105 = !{!"0x100\00self\0040\000", !42, !1, !34} ; [ DW_TAG_auto_variable ] [self] [line 40]
!106 = !MDLocation(line: 40, scope: !42)
!107 = !{!"llvm/tools/clang/test/CodeGenObjC/debug-info-block-captured-self.m", !""}
!108 = !{i32 1, !"Debug Info Version", i32 2}
!109 = !{!"0x102\0034\0032"} ; [ DW_TAG_expression ] [DW_OP_plus 32]
!110 = !{!"0x102\0034\0032"} ; [ DW_TAG_expression ] [DW_OP_plus 32]

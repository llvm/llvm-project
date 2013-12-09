; REQUIRES: object-emission

; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x03)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x01)
; CHECK-NOT: NULL
; CHECK: [[I:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK-NOT: NULL
; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name{{.*}}= "f1"
; CHECK: [[FUNC1:0x[0-9a-f]*]]:{{ *}}DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name{{.*}}= "f1"
; CHECK: NULL
; CHECK-NOT: NULL
; CHECK: [[FOO:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "foo"
; CHECK-NEXT: DW_AT_declaration
; CHECK-NOT: NULL
; CHECK: [[BAR:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "bar"
; CHECK: NULL
; CHECK: NULL
; CHECK: NULL

; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; This is a bug, it should be in F2 but it inherits the file from its
; enclosing scope
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x08)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x12)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x13)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FOO]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x14)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAR]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x15)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x16)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[I]]})
; CHECK-NOT: NULL
; CHECK: [[X:0x[0-9a-f]*]]:{{ *}}DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x18)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NEXT: DW_AT_name{{.*}}"X"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x19)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[X]]})
; CHECK-NEXT: DW_AT_name{{.*}}"Y"
; CHECK-NOT: NULL
; CHECK: DW_TAG_lexical_block
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x0f)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_imported_module
; Same bug as above, this should be F2, not F1
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x0b)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})

; CHECK: file_names[  [[F1]]]{{.*}}debug-info-namespace.cpp
; CHECK: file_names[  [[F2]]]{{.*}}foo.cpp

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is:
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; int i;
; void f1() { }
; void f1(int) { }
; struct foo;
; struct bar { };
; }
; using namespace B;
; }
;
; using namespace A;
;
; int func(bool b) {
;   if (b) {
;     using namespace A::B;
;     return i;
;   }
;   using namespace A;
;   using B::foo;
;   using B::bar;
;   using B::f1;
;   using B::i;
;   bar x;
;   namespace X = A;
;   namespace Y = X;
;   return i + X::B::i + Y::B::i;
; }

%"struct.A::B::bar" = type { i8 }

@_ZN1A1B1iE = global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1A1B2f1Ev() #0 {
entry:
  ret void, !dbg !41
}

; Function Attrs: nounwind uwtable
define void @_ZN1A1B2f1Ei(i32) #0 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %.addr}, metadata !42), !dbg !43
  ret void, !dbg !43
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %x = alloca %"struct.A::B::bar", align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.addr}, metadata !44), !dbg !45
  %0 = load i8* %b.addr, align 1, !dbg !46
  %tobool = trunc i8 %0 to i1, !dbg !46
  br i1 %tobool, label %if.then, label %if.end, !dbg !46

if.then:                                          ; preds = %entry
  %1 = load i32* @_ZN1A1B1iE, align 4, !dbg !47
  store i32 %1, i32* %retval, !dbg !47
  br label %return, !dbg !47

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata !{%"struct.A::B::bar"* %x}, metadata !48), !dbg !49
  %2 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %3 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %add = add nsw i32 %2, %3, !dbg !50
  %4 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %add1 = add nsw i32 %add, %4, !dbg !50
  store i32 %add1, i32* %retval, !dbg !50
  br label %return, !dbg !50

return:                                           ; preds = %if.end, %if.then
  %5 = load i32* %retval, !dbg !51
  ret i32 %5, !dbg !51
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!52}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !19, metadata !21, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug//usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/build/clang/debug"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !10, metadata !14}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"f1", metadata !"f1", metadata !"_ZN1A1B2f1Ev", i32 3, metadata !8, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN1A1B2f1Ev, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [f1]
!5 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/build/clang/debug"}
!6 = metadata !{i32 786489, metadata !5, metadata !7, metadata !"B", i32 1} ; [ DW_TAG_namespace ] [B] [line 1]
!7 = metadata !{i32 786489, metadata !1, null, metadata !"A", i32 3} ; [ DW_TAG_namespace ] [A] [line 3]
!8 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{null}
!10 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"f1", metadata !"f1", metadata !"_ZN1A1B2f1Ei", i32 4, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @_ZN1A1B2f1Ei, null, null, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{i32 786478, metadata !5, metadata !15, metadata !"func", metadata !"func", metadata !"_Z4funcb", i32 13, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i1)* @_Z4funcb, null, null, metadata !2, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [func]
!15 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!16 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{metadata !13, metadata !18}
!18 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786484, i32 0, metadata !6, metadata !"i", metadata !"i", metadata !"_ZN1A1B1iE", metadata !15, i32 2, metadata !13, i32 0, i32 1, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 2] [def]
!21 = metadata !{metadata !22, metadata !23, metadata !24, metadata !26, metadata !27, metadata !29, metadata !37, metadata !38, metadata !39, metadata !40}
!22 = metadata !{i32 786490, metadata !7, metadata !6, i32 8} ; [ DW_TAG_imported_module ]
!23 = metadata !{i32 786490, metadata !0, metadata !7, i32 11} ; [ DW_TAG_imported_module ]
!24 = metadata !{i32 786490, metadata !25, metadata !6, i32 15} ; [ DW_TAG_imported_module ]
!25 = metadata !{i32 786443, metadata !5, metadata !14, i32 14, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!26 = metadata !{i32 786490, metadata !14, metadata !7, i32 18} ; [ DW_TAG_imported_module ]
!27 = metadata !{i32 786440, metadata !14, metadata !28, i32 19} ; [ DW_TAG_imported_declaration ]
!28 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"foo", i32 5, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
!29 = metadata !{i32 786440, metadata !14, metadata !30, i32 20} ; [ DW_TAG_imported_declaration ]
!30 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"bar", i32 6, i64 8, i64 8, i32 0, i32 0, null, metadata !31, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 6, size 8, align 8, offset 0] [def] [from ]
!31 = metadata !{metadata !32}
!32 = metadata !{i32 786478, metadata !5, metadata !30, metadata !"bar", metadata !"bar", metadata !"", i32 6, metadata !33, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !36, i32 6} ; [ DW_TAG_subprogram ] [line 6] [bar]
!33 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !34, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{null, metadata !35}
!35 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !30} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!36 = metadata !{i32 786468}
!37 = metadata !{i32 786440, metadata !14, metadata !10, i32 21} ; [ DW_TAG_imported_declaration ]
!38 = metadata !{i32 786440, metadata !14, metadata !20, i32 22} ; [ DW_TAG_imported_declaration ]
!39 = metadata !{i32 786490, metadata !14, metadata !7, i32 24, metadata !"X"} ; [ DW_TAG_imported_module ]
!40 = metadata !{i32 786490, metadata !14, metadata !39, i32 25, metadata !"Y"} ; [ DW_TAG_imported_module ]
!41 = metadata !{i32 3, i32 0, metadata !4, null}
!42 = metadata !{i32 786689, metadata !10, metadata !"", metadata !15, i32 16777220, metadata !13, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 4]
!43 = metadata !{i32 4, i32 0, metadata !10, null}
!44 = metadata !{i32 786689, metadata !14, metadata !"b", metadata !15, i32 16777229, metadata !18, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 13]
!45 = metadata !{i32 13, i32 0, metadata !14, null}
!46 = metadata !{i32 14, i32 0, metadata !14, null}
!47 = metadata !{i32 16, i32 0, metadata !25, null}
!48 = metadata !{i32 786688, metadata !14, metadata !"x", metadata !15, i32 23, metadata !30, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [x] [line 23]
!49 = metadata !{i32 23, i32 0, metadata !14, null}
!50 = metadata !{i32 26, i32 0, metadata !14, null}
!51 = metadata !{i32 27, i32 0, metadata !14, null}
!52 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

; REQUIRES: default_triple, object-emission

; RUN: llvm-link %s %p/type-unique-simple-b.ll -S -o %t
; RUN: cat %t | FileCheck %s -check-prefix=LINK
; RUN: %llc_dwarf -filetype=obj -O0 < %t > %t2
; RUN: llvm-dwarfdump -v -debug-info %t2 | FileCheck %s

; Make sure the backend generates a single DIE and uses ref_addr.
; CHECK: 0x[[BASE:.*]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"Base"
; CHECK-NOT: DW_TAG_structure_type
; CHECK: 0x[[INT:.*]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}}"int"
; CHECK-NOT: DW_TAG_base_type

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[INT]]
; CHECK: DW_TAG_variable
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[BASE]]

; Make sure llvm-link only generates a single copy of the struct.
; LINK: DW_TAG_structure_type
; LINK-NOT: DW_TAG_structure_type
; Content of header files:
; struct Base {
;   int a;
; };
; Content of foo.cpp:
; 
; #include "a.hpp"
; void f(int a) {
;   Base t;
; }
; Content of bar.cpp:
; 
; #include "a.hpp"
; void f(int);
; void g(int a) {
;   Base t;
; }
; int main() {
;   f(0);
;   g(1);
;   return 0;
; }
; ModuleID = 'foo.cpp'

%struct.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1fi(i32 %a) #0 !dbg !10 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata %struct.Base* %t, metadata !17, metadata !DIExpression()), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "/Users/mren/c_testing/type_unique_air/simple")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "Base", line: 1, size: 32, align: 32, file: !5, elements: !6, identifier: "_ZTS4Base")
!5 = !DIFile(filename: "./a.hpp", directory: "/Users/mren/c_testing/type_unique_air/simple")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !5, scope: !4, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !11, type: !12, retainedNodes: !2)
!11 = !DIFile(filename: "foo.cpp", directory: "/Users/mren/c_testing/type_unique_air/simple")
!12 = !DISubroutineType(types: !13)
!13 = !{null, !8}
!14 = !{i32 2, !"Dwarf Version", i32 3}
!15 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !10, file: !11, type: !8)
!16 = !DILocation(line: 3, scope: !10)
!17 = !DILocalVariable(name: "t", line: 4, scope: !10, file: !11, type: !4)
!18 = !DILocation(line: 4, scope: !10)
!19 = !DILocation(line: 5, scope: !10)
!20 = !{i32 1, !"Debug Info Version", i32 3}

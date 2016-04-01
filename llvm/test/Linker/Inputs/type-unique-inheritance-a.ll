; CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "A"
; CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "Base"
; CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "B"
; CHECK-NOT: !DICompositeType(tag: DW_TAG_class_type
; Content of header files:
; 
; class Base;
; class A : Base {
;   int x;
; };
; 
; class A;
; class Base {
;   int b;
; };
; 
; class B {
;   int bb;
;   A *a;
; };
; Content of foo.cpp:
; 
; #include "b.hpp"
; #include "a.hpp"
; 
; void f(int a) {
;   A t;
; }
; Content of bar.cpp:
; 
; #include "b.hpp"
; #include "a.hpp"
; void g(int a) {
;   B t;
; }
; 
; void f(int);
; int main() {
;   A a;
;   f(0);
;   g(1);
;   return 0;
; }
; ModuleID = 'foo.cpp'

%class.A = type { %class.Base, i32 }
%class.Base = type { i32 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1fi(i32 %a) #0 !dbg !15 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %class.A, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata %class.A* %t, metadata !22, metadata !DIExpression()), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !25}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (http://llvm.org/git/clang.git f54e02f969d02d640103db73efc30c45439fceab) (http://llvm.org/git/llvm.git 284353b55896cb1babfaa7add7c0a363245342d2)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, subprograms: !14, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!2 = !{}
!3 = !{!4, !8}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 3, size: 64, align: 32, file: !5, elements: !6, identifier: "_ZTS1A")
!5 = !DIFile(filename: "./a.hpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!6 = !{!7, !13}
!7 = !DIDerivedType(tag: DW_TAG_inheritance, flags: DIFlagPrivate, scope: !"_ZTS1A", baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_class_type, name: "Base", line: 3, size: 32, align: 32, file: !9, elements: !10, identifier: "_ZTS4Base")
!9 = !DIFile(filename: "./b.hpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 4, size: 32, align: 32, flags: DIFlagPrivate, file: !9, scope: !"_ZTS4Base", baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 4, size: 32, align: 32, offset: 32, flags: DIFlagPrivate, file: !5, scope: !"_ZTS1A", baseType: !12)
!14 = !{!15}
!15 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !1, scope: !16, type: !17, variables: !2)
!16 = !DIFile(filename: "foo.cpp", directory: "/Users/mren/c_testing/type_unique_air/inher")
!17 = !DISubroutineType(types: !18)
!18 = !{null, !12}
!19 = !{i32 2, !"Dwarf Version", i32 2}
!20 = !DILocalVariable(name: "a", line: 5, arg: 1, scope: !15, file: !16, type: !12)
!21 = !DILocation(line: 5, scope: !15)
!22 = !DILocalVariable(name: "t", line: 6, scope: !15, file: !16, type: !4)
!23 = !DILocation(line: 6, scope: !15)
!24 = !DILocation(line: 7, scope: !15)
!25 = !{i32 1, !"Debug Info Version", i32 3}

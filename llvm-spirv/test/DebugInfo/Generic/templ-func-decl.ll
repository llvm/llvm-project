; REQUIRES: object-emission

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %t.ll | llvm-dwarfdump -v -debug-info - | FileCheck %s

; IR generated with `clang -Xclang -debug-info-kind=limited -emit-llvm -S` from the following code:
; class A {
; public:
;   template <int T> static int foo() { return T; }
; };
;
; int main() {
;   A::template foo<42>();
; }

; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name{{.*}}"foo<42>"

; CHECK: DW_TAG_template_value_parameter
; CHECK: DW_AT_type {{.*}} "int"
; CHECK: DW_AT_name {{.*}} "T"
; CHECK: DW_AT_const_value {{.*}} (42)

; ModuleID = 'templ-func-decl.cpp'
source_filename = "templ-func-decl.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

$_ZN1A3fooILi42EEEiv = comdat any

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %call = call i32 @_ZN1A3fooILi42EEEiv(), !dbg !10
  ret i32 0, !dbg !11
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @_ZN1A3fooILi42EEEiv() #1 comdat align 2 !dbg !12 {
entry:
  ret i32 42, !dbg !17
}

attributes #0 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "templ-func-decl.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 8.0.0 "}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !7, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 7, column: 3, scope: !6)
!11 = !DILocation(line: 8, column: 1, scope: !6)
!12 = distinct !DISubprogram(name: "foo<42>", linkageName: "_ZN1A3fooILi42EEEiv", scope: !13, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !15, declaration: !14, retainedNodes: !2)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue | DIFlagTrivial, elements: !2, identifier: "_ZTS1A")
!14 = !DISubprogram(name: "foo<42>", linkageName: "_ZN1A3fooILi42EEEiv", scope: !13, file: !1, line: 3, type: !7, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped | DIFlagStaticMember, isOptimized: false, templateParams: !15)
!15 = !{!16}
!16 = !DITemplateValueParameter(name: "T", type: !9, value: i32 42)
!17 = !DILocation(line: 3, column: 39, scope: !12)

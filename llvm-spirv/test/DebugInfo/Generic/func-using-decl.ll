; REQUIRES: object-emission

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple -O0 -filetype=obj < %t.ll > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Generated from the following source:
; extern void func_test1(int, int);
; namespace my_ns
; {
;   using ::func_test1;
; }
; int main ()
; {
;   return 0;
; }

; Ensure forward routine declarations with an associated using declaration
; are resolved properly (temporary node would trigger an assert).

; CHECK: DW_TAG_namespace
; CHECK: DW_AT_name {{.*}}"my_ns"
; CHECK: DW_TAG_imported_declaration
; CHECK: NULL
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}}"func_test1"
; CHECK: NULL

source_filename = "test/DebugInfo/Generic/func-using-decl.ll"

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !14 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !17
}

attributes #0 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !3, nameTableKind: None)
!1 = !DIFile(filename: "func-using-decl.cpp", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !1, line: 4)
!5 = !DINamespace(name: "my_ns", scope: null)
!6 = !DISubprogram(name: "func_test1", linkageName: "_Z10func_test1ii", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.0"}
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !15, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !DILocation(line: 8, column: 3, scope: !14)
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
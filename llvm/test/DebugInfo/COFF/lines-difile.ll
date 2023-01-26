; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; Test what happens when we have two DIFile entries with differing slashes.
; Make sure we only emit one file checksum entry.

; C++ source, compiled as:
; $ clang -S -emit-llvm -g -gcodeview ../t.cpp -o t.ll
; struct Foo { Foo(); };
; extern int global;
; Foo gy;
; void f() {
;   ++global;
;   ++global;
; }
; The relative path with a forward slash is important for creating TWO DIFile
; entries for the same file.

; ASM: .cv_file 1 "{{.*}}t.cpp" "8B89D3B180D6A1BC83E7126D5FD870C3" 1
; ASM-NOT: .cv_file 1 "{{.*}}t.cpp"

; OBJ:   SubSectionType: FileChecksums (0xF4)
; OBJ:   SubSectionSize: 0x18
; OBJ:   FileChecksum {
; OBJ:     Filename: C:\src\llvm-project\build\t.cpp (0x1)
; OBJ:     ChecksumSize: 0x10
; OBJ:     ChecksumKind: MD5 (0x1)
; OBJ:     ChecksumBytes: (8B 89 D3 B1 80 D6 A1 BC 83 E7 12 6D 5F D8 70 C3)
; OBJ:   }
; OBJ-NOT: FileChecksum {

; ModuleID = '../t.cpp'
source_filename = "../t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25508"

%struct.Foo = type { i8 }

@"\01?gy@@3UFoo@@A" = global %struct.Foo zeroinitializer, align 1, !dbg !0
@"\01?global@@3HA" = external global i32, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_t.cpp, ptr null }]

; Function Attrs: noinline uwtable
define internal void @"\01??__Egy@@YAXXZ"() #0 !dbg !18 {
entry:
  %call = call ptr @"\01??0Foo@@QEAA@XZ"(ptr @"\01?gy@@3UFoo@@A"), !dbg !21
  ret void, !dbg !21
}

declare ptr @"\01??0Foo@@QEAA@XZ"(ptr returned) unnamed_addr #1

; Function Attrs: noinline nounwind optnone uwtable
define void @"\01?f@@YAXXZ"() #2 !dbg !22 {
entry:
  %0 = load i32, ptr @"\01?global@@3HA", align 4, !dbg !23
  %inc = add nsw i32 %0, 1, !dbg !23
  store i32 %inc, ptr @"\01?global@@3HA", align 4, !dbg !23
  %1 = load i32, ptr @"\01?global@@3HA", align 4, !dbg !24
  %inc1 = add nsw i32 %1, 1, !dbg !24
  store i32 %inc1, ptr @"\01?global@@3HA", align 4, !dbg !24
  ret void, !dbg !25
}

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !26 {
entry:
  call void @"\01??__Egy@@YAXXZ"(), !dbg !28
  ret void
}

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gy", linkageName: "\01?gy@@3UFoo@@A", scope: !2, file: !6, line: 3, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "..\5Ct.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild\5Ctmp", checksumkind: CSK_MD5, checksum: "8b89d3b180d6a1bc83e7126d5fd870c3")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "../t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild\5Ctmp", checksumkind: CSK_MD5, checksum: "8b89d3b180d6a1bc83e7126d5fd870c3")
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !6, line: 1, size: 8, elements: !8, identifier: ".?AUFoo@@")
!8 = !{!9}
!9 = !DISubprogram(name: "Foo", scope: !7, file: !6, line: 1, type: !10, isLocal: false, isDefinition: false, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = !{i32 2, !"CodeView", i32 1}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 2}
!16 = !{i32 7, !"PIC Level", i32 2}
!17 = !{!"clang version 6.0.0 "}
!18 = distinct !DISubprogram(name: "??__Egy@@YAXXZ", scope: !6, file: !6, line: 3, type: !19, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocation(line: 3, column: 5, scope: !18)
!22 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !6, file: !6, line: 4, type: !19, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!23 = !DILocation(line: 5, column: 3, scope: !22)
!24 = !DILocation(line: 6, column: 3, scope: !22)
!25 = !DILocation(line: 7, column: 1, scope: !22)
!26 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !3, file: !3, type: !27, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !2, retainedNodes: !4)
!27 = !DISubroutineType(types: !4)
!28 = !DILocation(line: 0, scope: !26)

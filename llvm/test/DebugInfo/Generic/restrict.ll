; RUN: %llc_dwarf -dwarf-version=2 -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=CHECK --check-prefix=V2 %s
; RUN: %llc_dwarf -dwarf-version=3 -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=CHECK --check-prefix=V3 %s

; CHECK: DW_AT_name {{.*}} "dst"
; V2: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; V3: DW_AT_type {{.*}} {[[RESTRICT:0x.*]]}
; V3: [[RESTRICT]]: {{.*}}DW_TAG_restrict_type
; V3-NEXT: DW_AT_type {{.*}} {[[PTR:0x.*]]}
; CHECK: [[PTR]]: {{.*}}DW_TAG_pointer_type
; CHECK-NOT: DW_AT_type

; Generated with clang from:
; void foo(void* __restrict__ dst) {
; }


; Function Attrs: nounwind uwtable
define void @_Z3fooPv(i8* noalias %dst) #0 !dbg !4 {
entry:
  %dst.addr = alloca i8*, align 8
  store i8* %dst, i8** %dst.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %dst.addr, metadata !13, metadata !DIExpression()), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "restrict.c", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPv", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "restrict.c", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 1, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.5.0 "}
!13 = !DILocalVariable(name: "dst", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocation(line: 1, scope: !4)
!15 = !DILocation(line: 2, scope: !4)

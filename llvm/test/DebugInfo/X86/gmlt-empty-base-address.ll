; RUN: llc -function-sections -mtriple=x86_64-unknown-linux-gnu < %s -filetype=obj | llvm-dwarfdump -v -debug-info - | FileCheck %s

; CHECK: DW_AT_ranges [DW_FORM_rnglistx]   (indexed (0x0) rangelist = 0x00000010
; CHECK:                    [0x0000000000000000, 0x{{[0-9a-z]*}}) ".text._Z2f1v"
; CHECK:                    [0x0000000000000000, 0x{{[0-9a-z]*}}) ".text._Z2f2v")
; CHECK: DW_AT_addr_base [DW_FORM_sec_offset] (0x00000008)


; Function A
define dso_local void @_Z2f1v() #0 !dbg !7 {
  ret void, !dbg !9
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f2v() #0 !dbg !10 {
  ret void, !dbg !11
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0 (trunk 371665) (llvm/trunk 371681)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "gmlt-empty-base-address.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "74f7c574cd1ba04403967d02e757afeb")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (trunk 371665) (llvm/trunk 371681)"}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 1, scope: !7)
!10 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 4, column: 1, scope: !10)

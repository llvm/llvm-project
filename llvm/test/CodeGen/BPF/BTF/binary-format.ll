; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-readelf -x ".BTF" -x ".BTF.ext" - | FileCheck -check-prefixes=CHECK,CHECK-EL %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s | llvm-readelf -x ".BTF" -x ".BTF.ext" - | FileCheck -check-prefixes=CHECK,CHECK-EB %s

; Source code:
;   int f(int a) { return a; }
; Compilation flag:
;   clang -target bpf -O2 -g -gdwarf-5 -gembed-source -S -emit-llvm t.c

; Function Attrs: nounwind readnone
define dso_local i32 @f(i32 returned %a) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression()), !dbg !13
  ret i32 %a, !dbg !14
}

; CHECK:    '.BTF'
; CHECK-EL: 0x00000000 9feb0100 18000000 00000000 30000000
; CHECK-EL: 0x00000010 30000000 33000000 01000000 00000001
; CHECK-EL: 0x00000020 04000000 20000001 00000000 0100000d
; CHECK-EL: 0x00000030 01000000 05000000 01000000 07000000
; CHECK-EL: 0x00000040 0100000c 02000000 00696e74 00610066
; CHECK-EB: 0x00000000 eb9f0100 00000018 00000000 00000030
; CHECK-EB: 0x00000010 00000030 00000033 00000001 01000000
; CHECK-EB: 0x00000020 00000004 01000020 00000000 0d000001
; CHECK-EB: 0x00000030 00000001 00000005 00000001 00000007
; CHECK-EB: 0x00000040 0c000001 00000002 00696e74 00610066
; CHECK:    0x00000050 002e7465 7874002f 746d702f 742e6300
; CHECK:    0x00000060 696e7420 6628696e 74206129 207b2072
; CHECK:    0x00000070 65747572 6e20613b 207d00
; CHECK:    '.BTF.ext'
; CHECK-EL: 0x00000000 9feb0100 20000000 00000000 14000000
; CHECK-EL: 0x00000010 14000000 2c000000 40000000 00000000
; CHECK-EL: 0x00000020 08000000 09000000 01000000 00000000
; CHECK-EL: 0x00000030 03000000 10000000 09000000 02000000
; CHECK-EL: 0x00000040 00000000 0f000000 18000000 00040000
; CHECK-EL: 0x00000050 08000000 0f000000 18000000 10040000
; CHECK-EB: 0x00000000 eb9f0100 00000020 00000000 00000014
; CHECK-EB: 0x00000010 00000014 0000002c 00000040 00000000
; CHECK-EB: 0x00000020 00000008 00000009 00000001 00000000
; CHECK-EB: 0x00000030 00000003 00000010 00000009 00000002
; CHECK-EB: 0x00000040 00000000 0000000f 00000018 00000400
; CHECK-EB: 0x00000050 00000008 0000000f 00000018 00000410

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "1924f0d78deb326ceb76cd8e9f450775", source: "int f(int a) { return a; }\0A")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocation(line: 1, column: 11, scope: !7)
!14 = !DILocation(line: 1, column: 16, scope: !7)

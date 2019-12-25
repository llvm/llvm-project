; RUN: sed -e 's/nameTableKind: Default/nameTableKind: GNU/' %s | llc -mtriple=x86_64-pc-linux-gnu -filetype=obj | llvm-dwarfdump -v - | FileCheck --check-prefix=GPUB %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj < %s | llvm-dwarfdump -v - | FileCheck --check-prefix=NONE %s

; Generated from:
;   void f1();
;   inline __attribute__((always_inline)) void f2() {
;     f1();
;   }
;   void f3() {
;     f2();
;   }
;   $ clang++ -gmlt %s -emit-llvm -S

; GPUB: Compile Unit
; GPUB: DW_AT_GNU_pubnames

; GPUB: .debug_gnu_pubnames contents:
; GPUB-NEXT: unit_offset = 0x00000000
; GPUB-NEXT: Name
; GPUB-NEXT: "f2"
; GPUB-NEXT: "f3"

; GPUB: .debug_gnu_pubtypes contents:
; GPUB-NEXT: length = 0x0000000e version = 0x0002 unit_offset = 0x00000000
; GPUB-NEXT: Name

; NONE-NOT: .debug_pubnames contents:
; NONE-NOT: .debug_pubtypes contents:
; NONE-NOT: .debug_gnu_pubnames contents:
; NONE-NOT: .debug_gnu_pubtypes contents:


; Function Attrs: noinline uwtable
define void @_Z2f3v() #0 !dbg !7 {
entry:
  call void @_Z2f1v(), !dbg !9
  ret void, !dbg !12
}

declare void @_Z2f1v() #1

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 303768) (llvm/trunk 303774)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: Default)
!1 = !DIFile(filename: "gnu-public-names-gmlt.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 5.0.0 (trunk 303768) (llvm/trunk 303774)"}
!7 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 3, scope: !10, inlinedAt: !11)
!10 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = distinct !DILocation(line: 6, column: 3, scope: !7)
!12 = !DILocation(line: 7, column: 1, scope: !7)

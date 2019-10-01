; RUN: llc %s -mtriple=i386-unknown-linux-gnu -filetype=asm -o - | FileCheck %s

; CHECK:   .section .debug_addr
; CHECK-NEXT:   .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
; CHECK-NEXT: .Ldebug_addr_start0:
; CHECK-NEXT:   .short  5 # DWARF version number
; CHECK-NEXT:   .byte   4 # Address size
; CHECK-NEXT:   .byte   0 # Segment selector size
; CHECK-NEXT: .Laddr_table_base0:
; CHECK-NEXT:   .long   .Lfunc_begin0
; CHECK-NEXT: .Ldebug_addr_end0:
 
; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !7 {
entry:
  ret void, !dbg !10
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 350004) (llvm/trunk 350008)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 350004) (llvm/trunk 350008)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 2, column: 1, scope: !7)

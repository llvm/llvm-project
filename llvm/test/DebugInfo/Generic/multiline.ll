; RUN: llc -filetype=asm -asm-verbose=0 -O0 < %s | FileCheck %s
; RUN: llc -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=line - | FileCheck %s --check-prefix=INT

; Check that the assembly output properly handles is_stmt changes. And since
; we're testing anyway, check the integrated assembler too.

; Generated with clang from multiline.c:
; void f1();
; void f2() {
;   f1(); f1(); f1();
;   f1(); f1(); f1();
; }


; CHECK: .loc	1 2 0{{$}}
; CHECK-NOT: .loc{{ }}
; CHECK: .loc	1 3 3 prologue_end{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 3 9 is_stmt 0{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 3 15{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 4 3 is_stmt 1{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 4 9 is_stmt 0{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 4 15{{$}}
; CHECK-NOT: .loc
; CHECK: .loc	1 5 1 is_stmt 1{{$}}

; INT: {{^}}Address
; INT: -----
; INT-NEXT: 2 0 1 0 0 is_stmt{{$}}
; INT-NEXT: 3 3 1 0 0 is_stmt prologue_end{{$}}
; INT-NEXT: 3 9 1 0 0 {{$}}
; INT-NEXT: 3 15 1 0 0 {{$}}
; INT-NEXT: 4 3 1 0 0 is_stmt{{$}}
; INT-NEXT: 4 9 1 0 0 {{$}}
; INT-NEXT: 4 15 1 0 0 {{$}}
; INT-NEXT: 5 1 1 0 0 is_stmt{{$}}


; Function Attrs: nounwind uwtable
define void @f2() #0 !dbg !4 {
entry:
  call void (...) @f1(), !dbg !11
  call void (...) @f1(), !dbg !12
  call void (...) @f1(), !dbg !13
  call void (...) @f1(), !dbg !14
  call void (...) @f1(), !dbg !15
  call void (...) @f1(), !dbg !16
  ret void, !dbg !17
}

declare void @f1(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk 225000) (llvm/trunk 224999)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "multiline.c", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f2", line: 2, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "multiline.c", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.6.0 (trunk 225000) (llvm/trunk 224999)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 3, column: 9, scope: !4)
!13 = !DILocation(line: 3, column: 15, scope: !4)
!14 = !DILocation(line: 4, column: 3, scope: !4)
!15 = !DILocation(line: 4, column: 9, scope: !4)
!16 = !DILocation(line: 4, column: 15, scope: !4)
!17 = !DILocation(line: 5, column: 1, scope: !4)

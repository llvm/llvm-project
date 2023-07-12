; RUN: llc -mtriple=mips-linux-gnu -filetype=asm -asm-verbose=0 -O0 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=mips-linux-gnu -filetype=obj -O0 < %s | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefix=INT

; Mips used to generate 'jumpy' debug line info around calls. The address
; calculation for each call to f1() would share the same line info so it would
; emit output of the form:
;   .loc $first_call_location
;   .. address calculation ..
;   .. function call ..
;   .. address calculation ..
;   .loc $second_call_location
;   .. function call ..
;   .loc $first_call_location
;   .. address calculation ..
;   .loc $third_call_location
;   .. function call ..
;   ...
; which would cause confusing stepping behaviour for the end user.
;
; This test checks that we emit more user friendly debug line info of the form:
;   .loc $first_call_location
;   .. address calculation ..
;   .. function call ..
;   .loc $second_call_location
;   .. address calculation ..
;   .. function call ..
;   .loc $third_call_location
;   .. address calculation ..
;   .. function call ..
;   ...
;
; Generated with clang from fn-call-line.c:
; void f1();
; void f2() {
;   f1();
;   f1();
; }

; CHECK: .loc	1 3 3
; CHECK-NOT: .loc
; CHECK: %call16(f1) 
; CHECK-NOT: .loc
; CHECK: .loc	1 4 3
; CHECK-NOT: .loc
; CHECK: %call16(f1) 

; INT: {{^}}Address
; INT: -----
; INT-NEXT: 2 0 1 0 0 0 is_stmt{{$}}
; INT-NEXT: 3 3 1 0 0 0 is_stmt prologue_end{{$}}
; INT-NEXT: 4 3 1 0 0 0 is_stmt{{$}}


; Function Attrs: nounwind uwtable
define void @f2() #0 !dbg !4 {
entry:
  call void (...) @f1(), !dbg !11
  call void (...) @f1(), !dbg !12
  ret void, !dbg !13
}

declare void @f1(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0 (trunk 226641)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "fn-call-line.c", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f2", line: 2, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "fn-call-line.c", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.7.0 (trunk 226641)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 4, column: 3, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)

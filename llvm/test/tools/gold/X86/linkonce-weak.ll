; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/linkonce-weak.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t2.o %t.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

define linkonce_odr void @f() !dbg !4 {
  ret void, !dbg !10
}

; Test that we get a weak_odr regardless of the order of the files
; CHECK: define weak_odr void @f()

; Test that we only get a single DISubprogram for @f
; CHECK: !DISubprogram(name: "f"
; CHECK-NOT: !DISubprogram(name: "f"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "linkonce-weak.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)"}
!10 = !DILocation(line: 2, column: 1, scope: !4)

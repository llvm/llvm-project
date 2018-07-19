; RUN: llc < %s -stop-before=expand-isel-pseudos -o - | FileCheck %s

source_filename = "foo.c"
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "ppc32"

; Verify that, when handling split-up integers, the
; transferring of debug info takes the endianness
; into consideration.
;
; The fragment expression at offset 0 should correspond
; to the high part of the value on big-endian targets.

; This basis of this ll file was created by running:
;   clang --target=powerpc -O1 -S -g -emit-llvm foo.c
;
; with foo.c being the program:
;   unsigned long long foo(void);
;   void bar() {
;     volatile unsigned long long result = foo();
;   }
;
; This file is a slight tweak of that output, with irrelevant
; lifetime intrinsics, metadata, and debug info being removed.

; CHECK: [[DL:![0-9]+]] = !DILocalVariable(name: "result"
;
; High 32 bits in R3, low 32 bits in R4
; CHECK: %0:gprc = COPY %r3
; CHECK: DBG_VALUE debug-use %0, debug-use %noreg, [[DL]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK: %1:gprc = COPY %r4
; CHECK: DBG_VALUE debug-use %1, debug-use %noreg, [[DL]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
define void @bar() local_unnamed_addr #0 !dbg !6 {
  %1 = alloca i64, align 8
  %2 = tail call i64 @foo()
  tail call void @llvm.dbg.value(metadata i64 %2, metadata !10, metadata !DIExpression()), !dbg !13
  store volatile i64 %2, i64* %1, align 8
  ret void
}

declare i64 @foo() local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 6.0.0"}
!6 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, variables: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10}
!10 = !DILocalVariable(name: "result", scope: !6, file: !1, line: 3, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !12)
!12 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!13 = !DILocation(line: 3, column: 31, scope: !6)

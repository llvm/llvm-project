; RUN: llc -march=mips -mcpu=mips32r2 -O0 -filetype=obj <%s | \
; RUN:    llvm-dwarfdump -debug-dump=all - | FileCheck %s -check-prefix=F2
; RUN: llc -march=mips -mcpu=mips32r2 -O0 -filetype=obj <%s | \
; RUN:    llvm-dwarfdump -debug-dump=all - | FileCheck %s -check-prefix=F3

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @foo(i32*)

; void foo(int *);
;
; int f2(int a, int b) {
;   int c __attribute__((aligned(16))) = a + b;
;   foo(&c);
;   return c;
; }
;
; int *f3(int a, int b) {
;   int c __attribute__((aligned(16))) = a + b;
;   int *w = alloca(c);
;   foo(&c);
;   return w;
; }

; c -> DW_OP_breg29(r29): 16
; F2: DW_AT_location [DW_FORM_exprloc]      (<0x2> 8d 10 )
; F2: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x00000065] = "c")

; Function Attrs: nounwind
define i32 @f2(i32 signext %a, i32 signext %b) !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 16
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !16), !dbg !17
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !18, metadata !16), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %c, metadata !20, metadata !16), !dbg !21
  %0 = load i32, i32* %a.addr, align 4, !dbg !22
  %1 = load i32, i32* %b.addr, align 4, !dbg !23
  %add = add nsw i32 %0, %1, !dbg !24
  store i32 %add, i32* %c, align 16, !dbg !21
  call void @foo(i32* %c), !dbg !25
  %2 = load i32, i32* %c, align 16, !dbg !26
  ret i32 %2, !dbg !27
}

; c -> DW_OP_breg23(r23): 16
; F3: DW_AT_location [DW_FORM_exprloc]      (<0x2> 87 10 )
; F3: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x00000065] = "c")

define i32* @f3(i32 signext %a, i32 signext %b) !dbg !8 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 16
  %w = alloca i32*, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !28, metadata !16), !dbg !29
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !30, metadata !16), !dbg !31
  call void @llvm.dbg.declare(metadata i32* %c, metadata !32, metadata !16), !dbg !33
  %0 = load i32, i32* %a.addr, align 4, !dbg !34
  %1 = load i32, i32* %b.addr, align 4, !dbg !35
  %add = add nsw i32 %0, %1, !dbg !36
  store i32 %add, i32* %c, align 16, !dbg !33
  call void @llvm.dbg.declare(metadata i32** %w, metadata !37, metadata !DIExpression(DW_OP_deref)), !dbg !38
  %2 = load i32, i32* %c, align 16, !dbg !39
  %3 = alloca i8, i32 %2, !dbg !40
  %4 = bitcast i8* %3 to i32*, !dbg !40
  store i32* %4, i32** %w, align 4, !dbg !38
  call void @foo(i32* %c), !dbg !41
  %5 = load i32*, i32** %w, align 4, !dbg !42
  ret i32* %5, !dbg !43
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251783) (llvm/trunk 251781)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/vk/repos/tmp/dwarf")
!2 = !{}
!4 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 20, type: !5, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 27, type: !9, isLocal: false, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !7, !7}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.8.0 (trunk 251783) (llvm/trunk 251781)"}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 20, type: !7)
!16 = !DIExpression()
!17 = !DILocation(line: 20, column: 12, scope: !4)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 20, type: !7)
!19 = !DILocation(line: 20, column: 19, scope: !4)
!20 = !DILocalVariable(name: "c", scope: !4, file: !1, line: 21, type: !7)
!21 = !DILocation(line: 21, column: 7, scope: !4)
!22 = !DILocation(line: 21, column: 40, scope: !4)
!23 = !DILocation(line: 21, column: 44, scope: !4)
!24 = !DILocation(line: 21, column: 42, scope: !4)
!25 = !DILocation(line: 22, column: 3, scope: !4)
!26 = !DILocation(line: 23, column: 10, scope: !4)
!27 = !DILocation(line: 23, column: 3, scope: !4)
!28 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 27, type: !7)
!29 = !DILocation(line: 27, column: 13, scope: !8)
!30 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 27, type: !7)
!31 = !DILocation(line: 27, column: 20, scope: !8)
!32 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 28, type: !7)
!33 = !DILocation(line: 28, column: 7, scope: !8)
!34 = !DILocation(line: 28, column: 40, scope: !8)
!35 = !DILocation(line: 28, column: 44, scope: !8)
!36 = !DILocation(line: 28, column: 42, scope: !8)
!37 = !DILocalVariable(name: "w", scope: !8, file: !1, line: 29, type: !11)
!38 = !DILocation(line: 29, column: 8, scope: !8)
!39 = !DILocation(line: 29, column: 19, scope: !8)
!40 = !DILocation(line: 29, column: 12, scope: !8)
!41 = !DILocation(line: 30, column: 3, scope: !8)
!42 = !DILocation(line: 31, column: 10, scope: !8)
!43 = !DILocation(line: 31, column: 3, scope: !8)

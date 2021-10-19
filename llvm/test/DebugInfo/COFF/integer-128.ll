; RUN: llc < %s | FileCheck %s --check-prefix=ASM

; // C++ source to regenerate:
; enum class uns : __uint128_t { unsval = __uint128_t(1) << 64 };
; uns t1() { return uns::unsval; }
; enum class sig : __int128 { sigval = -(__int128(1) << 64) };
; sig t2() { return sig::sigval; }
; struct test {
;   static const __uint128_t u128 = __uint128_t(1) << 64;
;   static const __int128    s128 = -(__int128(1) << 64);
; };
; test t3() { return test(); }
; 
; $ clang a.cpp -S -emit-llvm -g -gcodeview

; ------------------------------------------------------------------------------

; ASM-LABEL: .long   241                             # Symbol subsection for globals
;
; ASM-LABEL: .short	4359                            # Record kind: S_CONSTANT
; ASM-NEXT:  .long	4110                            # Type
; ASM-NEXT:  .byte	0x0a, 0x80, 0xff, 0xff          # Value
; ASM-NEXT:  .byte	0xff, 0xff, 0xff, 0xff
; ASM-NEXT:  .byte	0xff, 0xff
; ASM-NEXT:  .asciz	"test::u128"                    # Name
; ASM-NEXT:  .p2align	2
;
; ASM-LABEL: .short	4359                            # Record kind: S_CONSTANT
; ASM-NEXT:  .long	4111                            # Type
; ASM-NEXT:  .byte	0x09, 0x80, 0x00, 0x00          # Value
; ASM-NEXT:  .byte	0x00, 0x00, 0x00, 0x00
; ASM-NEXT:  .byte	0x00, 0x80
; ASM-NEXT:  .asciz	"test::s128"                    # Name
; ASM-NEXT:  .p2align	2
;
; ASM-LABEL: .short	0x1203                          # Record kind: LF_FIELDLIST
; ASM-NEXT:  .short	0x1502                          # Member kind: Enumerator ( LF_ENUMERATE )
; ASM-NEXT:  .short	0x3                             # Attrs: Public
; ASM-NEXT:  .short	0x800a
; ASM-NEXT:  .quad	0xffffffffffffffff              # EnumValue
; ASM-NEXT:  .asciz	"unsval"                        # Name
;
; ASM-LABEL: .short	0x1203                          # Record kind: LF_FIELDLIST
; ASM-NEXT:  .short	0x1502                          # Member kind: Enumerator ( LF_ENUMERATE )
; ASM-NEXT:  .short	0x3                             # Attrs: Public
; ASM-NEXT:  .short	0x800a
; ASM-NEXT:  .quad	0xffffffffffffffff              # EnumValue
; ASM-NEXT:  .asciz	"sigval"                        # Name

; ------------------------------------------------------------------------------

; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

%struct.test = type { i8 }

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <2 x i64> @_Z2t1v() #0 !dbg !23 {
entry:
  %retval = alloca i128, align 16
  store i128 18446744073709551616, i128* %retval, align 16, !dbg !27
  %0 = bitcast i128* %retval to <2 x i64>*, !dbg !27
  %1 = load <2 x i64>, <2 x i64>* %0, align 16, !dbg !27
  ret <2 x i64> %1, !dbg !27
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local <2 x i64> @_Z2t2v() #0 !dbg !28 {
entry:
  %retval = alloca i128, align 16
  store i128 -18446744073709551616, i128* %retval, align 16, !dbg !31
  %0 = bitcast i128* %retval to <2 x i64>*, !dbg !31
  %1 = load <2 x i64>, <2 x i64>* %0, align 16, !dbg !31
  ret <2 x i64> %1, !dbg !31
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local i8 @_Z2t3v() #1 !dbg !32 {
entry:
  %retval = alloca %struct.test, align 1
  %coerce.dive = getelementptr inbounds %struct.test, %struct.test* %retval, i32 0, i32 0, !dbg !41
  %0 = load i8, i8* %coerce.dive, align 1, !dbg !41
  ret i8 %0, !dbg !41
}

attributes #0 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !14, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "b37f4034fd610917975e9c5ff097fa6b")
!2 = !{!3, !10}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "uns", file: !4, line: 4, baseType: !5, size: 128, flags: DIFlagEnumClass, elements: !8, identifier: "_ZTS3uns")
!4 = !DIFile(filename: "a.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "b37f4034fd610917975e9c5ff097fa6b")
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint128_t", file: !6, baseType: !7)
!6 = !DIFile(filename: "a.cpp", directory: ".")
!7 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
!8 = !{!9}
!9 = !DIEnumerator(name: "unsval", value: 18446744073709551616, isUnsigned: true)
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "sig", file: !4, line: 7, baseType: !11, size: 128, flags: DIFlagEnumClass, elements: !12, identifier: "_ZTS3sig")
!11 = !DIBasicType(name: "__int128", size: 128, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DIEnumerator(name: "sigval", value: -18446744073709551616)
!14 = !{!15, !17}
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "unsval", scope: !0, file: !4, line: 4, type: !3, isLocal: true, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "sigval", scope: !0, file: !4, line: 7, type: !10, isLocal: true, isDefinition: true)
!19 = !{i32 2, !"CodeView", i32 1}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 2}
!22 = !{!"clang version 13.0.0"}
!23 = distinct !DISubprogram(name: "t1", linkageName: "_Z2t1v", scope: !4, file: !4, line: 5, type: !24, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!24 = !DISubroutineType(types: !25)
!25 = !{!3}
!26 = !{}
!27 = !DILocation(line: 5, column: 12, scope: !23)
!28 = distinct !DISubprogram(name: "t2", linkageName: "_Z2t2v", scope: !4, file: !4, line: 8, type: !29, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!29 = !DISubroutineType(types: !30)
!30 = !{!10}
!31 = !DILocation(line: 8, column: 12, scope: !28)
!32 = distinct !DISubprogram(name: "t3", linkageName: "_Z2t3v", scope: !4, file: !4, line: 14, type: !33, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
!33 = !DISubroutineType(types: !34)
!34 = !{!35}
!35 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "test", file: !4, line: 10, size: 8, flags: DIFlagTypePassByValue, elements: !36, identifier: "_ZTS4test")
!36 = !{!37, !39}
!37 = !DIDerivedType(tag: DW_TAG_member, name: "u128", scope: !35, file: !4, line: 11, baseType: !38, flags: DIFlagStaticMember, extraData: i128 18446744073709551616)
!38 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "s128", scope: !35, file: !4, line: 12, baseType: !40, flags: DIFlagStaticMember, extraData: i128 -18446744073709551616)
!40 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!41 = !DILocation(line: 14, column: 13, scope: !32)

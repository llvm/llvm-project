; RUN: llc -mtriple=aarch64-windows -filetype=obj -o - %s | \
; RUN: llvm-readobj --codeview - | FileCheck %s

; ModuleID = 'a.c'
source_filename = "a.c"
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--windows-msvc18.0.0"

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 1, !dbg !11
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "/", checksumkind: CSK_MD5, checksum: "12345678901234567890123456789012")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 3, scope: !7)

; CHECK: Format: COFF-ARM64
; CHECK: Arch: aarch64
; CHECK: AddressSize: 64bit
; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (5)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 0
; CHECK:     Arguments [
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: int (0x74)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 0
; CHECK:     ArgListType: () (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: int () (0x1001)
; CHECK:     Name: main
; CHECK:   }
; CHECK: ]
; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (4)
; CHECK:   Magic: 0x4
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     Compile3Sym {
; CHECK:       Kind: S_COMPILE3 (0x113C)
; CHECK:       Language: C (0x0)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalProcIdSym {
; CHECK:       Kind: S_GPROC32_ID (0x1147)
; CHECK:       PtrParent: 0x0
; CHECK:       PtrEnd: 0x0
; CHECK:       PtrNext: 0x0
; CHECK:       CodeSize: 0x14
; CHECK:       DbgStart: 0x0
; CHECK:       DbgEnd: 0x0
; CHECK:       FunctionType: main (0x1002)
; CHECK:       CodeOffset: main+0x0
; CHECK:       Segment: 0x0
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       DisplayName: main
; CHECK:       LinkageName: main
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:       Kind: S_PROC_ID_END (0x114F)
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: Lines (0xF2)
; CHECK:     LinkageName: main
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: FileChecksums (0xF4)
; CHECK:     FileChecksum {
; CHECK:       ChecksumSize: 0x10
; CHECK:       ChecksumKind: MD5 (0x1)
; CHECK:       ChecksumBytes: (12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 12)
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: StringTable (0xF3)
; CHECK:   ]
; CHECK:   FunctionLineTable [
; CHECK:     LinkageName: main
; CHECK:     Flags: 0x1
; CHECK:     CodeSize: 0x14
; CHECK:     FilenameSegment [
; CHECK:       +0x0 [
; CHECK:         LineNumberStart: 1
; CHECK:         LineNumberEndDelta: 0
; CHECK:         IsStatement: No
; CHECK:         ColStart: 0
; CHECK:         ColEnd: 0
; CHECK:       ]
; CHECK:       +0x8 [
; CHECK:         LineNumberStart: 2
; CHECK:         LineNumberEndDelta: 0
; CHECK:         IsStatement: No
; CHECK:         ColStart: 3
; CHECK:         ColEnd: 0
; CHECK:       ]
; CHECK:     ]
; CHECK:   ]
; CHECK: ]

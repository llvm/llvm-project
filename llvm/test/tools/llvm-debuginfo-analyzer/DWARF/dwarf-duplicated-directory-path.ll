; REQUIRES: x86-registered-target

; When the 'filename' field in DIFile entry contains the full pathname, such as:
;
; !1 = !DIFile(filename: "path/test.cpp", directory: "path")
;
; llvm-debuginfo-analyzer displays incorrect information for the directories
; extracted from the '.debug_line' section, when using '--attribute=directories'
;
; llvm-debuginfo-analyzer --attribute=directories --print=scopes test.o
;
; {CompileUnit} 'test.cpp'
;   {Directory} 'path/path'  <------- Duplicated 'path'

; The test case was produced by the following steps:
; // test.cpp
; void foo() {
; }

; 1) clang++ --target=x86_64-pc-linux-gnu -Xclang -disable-O0-optnone
;            -Xclang -disable-llvm-passes -fno-discard-value-names
;            -emit-llvm -S -g -O0 test.cpp -o test.ll
;
; 2) Manually adding directory information to the DIFile 'filename' field:
;
; !1 = !DIFile(filename: "test.cpp", directory: "path")
; -->
; !1 = !DIFile(filename: "path/test.cpp", directory: "path")

; RUN: llc --filetype=obj %p/dwarf-duplicated-directory-path.ll \
; RUN:                    -o %t.dwarf-duplicated-directory-path.o

; RUN: llvm-debuginfo-analyzer --attribute=directories,files               \
; RUN:                         --print=scopes                              \
; RUN:                         %t.dwarf-duplicated-directory-path.o 2>&1 | \
; RUN: FileCheck --strict-whitespace -check-prefix=ONE %s

; ONE: Logical View:
; ONE-NEXT:           {File} '{{.*}}dwarf-duplicated-directory-path.ll{{.*}}'
; ONE-EMPTY:
; ONE-NEXT:             {CompileUnit} 'test.cpp'
; ONE-NEXT:               {Directory} 'path'
; ONE-NEXT:               {File} 'test.cpp'
; ONE-NEXT:     1         {Function} extern not_inlined 'foo' -> 'void'

source_filename = "test.cpp"
target triple = "x86_64-pc-linux-gnu"

define dso_local void @_Z3foov() !dbg !10 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "path/test.cpp", directory: "path")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 2, column: 1, scope: !10)

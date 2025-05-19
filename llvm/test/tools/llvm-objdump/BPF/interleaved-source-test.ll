; REQUIRES: bpf-registered-target

;; Verify that llvm-objdump can use .BTF.ext to extract line number
;; information in disassembly when DWARF is not available.

;; The 'sed' part is needed because llc would look for source file in
;; order to embed line info when BPF is compiled.

; RUN: sed -e "s,SRC_COMPDIR,%/p/Inputs,g" %s > %t.ll

;; First, check bpfel (little endian):
;; - compile %t.o
;; - check llvm-objdump output when both DWARF and BTF are present
;; - strip debug info from %t.o
;; - make sure that there are BTF but no DWARF sections in %t.o
;; - check llvm-objdump output when only BTF is present

; RUN: llc --mtriple bpfel -mcpu=v1 %t.ll --filetype=obj -o %t
; RUN: llvm-objdump --no-show-raw-insn -S %t | FileCheck %s
; RUN: llvm-strip --strip-debug %t
; RUN: llvm-objdump --section-headers %t \
; RUN:   | FileCheck --implicit-check-not=.debug_ --check-prefix=SECTIONS %s
; RUN: llvm-objdump --no-show-raw-insn -S %t | FileCheck %s

;; Next, check bpfeb (big endian):

; RUN: llc --mtriple bpfeb -mcpu=v1 %t.ll --filetype=obj -o %t
; RUN: llvm-strip --strip-debug %t
; RUN: llvm-objdump --no-show-raw-insn -S %t | FileCheck %s

;; Test case adapted from output of the following command:
;;
;;  clang -g --target=bpf -emit-llvm -S ./Inputs/test.c
;;
;; DIFile::directory is changed to SRC_COMPDIR.

; SECTIONS: .BTF
; SECTIONS: .BTF.ext

;; Check inlined source code in disassembly:

; CHECK:      Disassembly of section .text:
; CHECK-EMPTY:
; CHECK-NEXT: [[#%x,]] <foo>:
; CHECK-NEXT: ;   consume(1);
; CHECK-NEXT:        0:	r1 = 0x1
; CHECK-NEXT:        1:	call -0x1
; CHECK-NEXT: ;   consume(2);
; CHECK-NEXT:        2:	r1 = 0x2
; CHECK-NEXT:        3:	call -0x1
; CHECK-NEXT: ; }
; CHECK-NEXT:        4:	exit
; CHECK-EMPTY:
; CHECK-NEXT: [[#%x,]] <bar>:
; CHECK-NEXT: ;   consume(3);
; CHECK-NEXT:        5:	r1 = 0x3
; CHECK-NEXT:        6:	call -0x1
; CHECK-NEXT: ; }
; CHECK-NEXT:        7:	exit
; CHECK-EMPTY:
; CHECK-NEXT: Disassembly of section a:
; CHECK-EMPTY:
; CHECK-NEXT: [[#%x,]] <buz>:
; CHECK-NEXT: ;   consume(4);
; CHECK-NEXT:        0:	r1 = 0x4
; CHECK-NEXT:        1:	call -0x1
; CHECK-NEXT: ; }
; CHECK-NEXT:        2:	exit
; CHECK-EMPTY:
; CHECK-NEXT: Disassembly of section b:
; CHECK-EMPTY:
; CHECK-NEXT: [[#%x,]] <quux>:
; CHECK-NEXT: ;   consume(5);
; CHECK-NEXT:        0:	r1 = 0x5
; CHECK-NEXT:        1:	call -0x1
; CHECK-NEXT: ; }
; CHECK-NEXT:        2:	exit

; Function Attrs: noinline nounwind optnone
define dso_local void @foo() #0 !dbg !7 {
entry:
  %call = call i32 @consume(i32 noundef 1), !dbg !11
  %call1 = call i32 @consume(i32 noundef 2), !dbg !12
  ret void, !dbg !13
}

declare dso_local i32 @consume(i32 noundef) #1

; Function Attrs: noinline nounwind optnone
define dso_local void @bar() #0 !dbg !14 {
entry:
  %call = call i32 @consume(i32 noundef 3), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind optnone
define dso_local void @buz() #0 section "a" !dbg !17 {
entry:
  %call = call i32 @consume(i32 noundef 4), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: noinline nounwind optnone
define dso_local void @quux() #0 section "b" !dbg !20 {
entry:
  %call = call i32 @consume(i32 noundef 5), !dbg !21
  ret void, !dbg !22
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 81674c88f80fa7d9c55d4aee945f844b67f03267)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "SRC_COMPDIR", checksumkind: CSK_MD5, checksum: "292d67837b080844462efb2a6b004f09")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 81674c88f80fa7d9c55d4aee945f844b67f03267)"}
!7 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DIFile(filename: "test.c", directory: "SRC_COMPDIR", checksumkind: CSK_MD5, checksum: "292d67837b080844462efb2a6b004f09")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 4, column: 3, scope: !7)
!12 = !DILocation(line: 5, column: 3, scope: !7)
!13 = !DILocation(line: 6, column: 1, scope: !7)
!14 = distinct !DISubprogram(name: "bar", scope: !8, file: !8, line: 8, type: !9, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 9, column: 3, scope: !14)
!16 = !DILocation(line: 10, column: 1, scope: !14)
!17 = distinct !DISubprogram(name: "buz", scope: !8, file: !8, line: 13, type: !9, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DILocation(line: 14, column: 3, scope: !17)
!19 = !DILocation(line: 15, column: 1, scope: !17)
!20 = distinct !DISubprogram(name: "quux", scope: !8, file: !8, line: 18, type: !9, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocation(line: 19, column: 3, scope: !20)
!22 = !DILocation(line: 20, column: 1, scope: !20)

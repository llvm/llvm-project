; RUN: rm -rf %t && mkdir -p %t
; RUN: llc --filetype=obj -cas-friendly-debug-info %s -o %t/cas-friendly.o 
; RUN: llvm-dwarfdump %t/cas-friendly.o --debug-line | FileCheck %s
; CHECK: Address            Line   Column File   ISA Discriminator OpIndex Flags
; CHECK-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; CHECK-NOT: 0x0000000000000000      0      0      1   0             0       0  is_stmt
; CHECK-NEXT: 0x0000000000000000      0      0      0   0             0       0  is_stmt
; REQUIRES: arm-registered-target

define i32 @_Z3fooi(i32 noundef %x) #0 !dbg !9 {
entry:
  %x.addr = alloca i32
  %0 = load i32, ptr %x.addr, align 4
  %add = add nsw i32 %0, 1, !dbg !18
  ret i32 %add
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !7}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug, sysroot: "/")
!1 = !DIFile(filename: "/tmp/a.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "80e82b6b66c04b9cfb438eb7c6672107")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2,!"Debug Info Version", i32 3}
!7 = !{i32 7, !"frame-pointer", i32 1}
!9 = distinct !DISubprogram(file: !10, type: !11, unit: !0, retainedNodes: !14)
!10 = !DIFile(filename: "/tmp/a.cpp", directory: "", checksumkind: CSK_MD5, checksum: "80e82b6b66c04b9cfb438eb7c6672107")
!11 = !DISubroutineType(types: !12)
!12 = !{}
!14 = !{}
!18 = !DILocation(scope: !9)

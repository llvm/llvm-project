; RUN: llc -O2 -mtriple=riscv64 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ZBB
; RUN: llc -O2 -mtriple=riscv64 -mattr=-zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOZBB
; RUN: llc -O0 -mtriple=riscv64 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -o /dev/null

define i64 @byte_bit_0(i64 %x) {
; ZBB-LABEL: byte_bit_0:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_0:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 72340172838076673
  %l = shl i64 %m, 8
  %r = sub i64 %l, %m
  ret i64 %r
}

define i64 @byte_bit_1(i64 %x) {
; ZBB-LABEL: byte_bit_1:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_1:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 144680345676153346
  %l = shl i64 %m, 7
  %h = lshr i64 %m, 1
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_2(i64 %x) {
; ZBB-LABEL: byte_bit_2:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_2:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 289360691352306692
  %l = shl i64 %m, 6
  %h = lshr i64 %m, 2
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_3(i64 %x) {
; ZBB-LABEL: byte_bit_3:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_3:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 578721382704613384
  %l = shl i64 %m, 5
  %h = lshr i64 %m, 3
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_4(i64 %x) {
; ZBB-LABEL: byte_bit_4:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_4:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 1157442765409226768
  %l = shl i64 %m, 4
  %h = lshr i64 %m, 4
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_5(i64 %x) {
; ZBB-LABEL: byte_bit_5:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_5:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 2314885530818453536
  %l = shl i64 %m, 3
  %h = lshr i64 %m, 5
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_6(i64 %x) {
; ZBB-LABEL: byte_bit_6:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_6:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 4629771061636907072
  %l = shl i64 %m, 2
  %h = lshr i64 %m, 6
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @byte_bit_7(i64 %x) {
; ZBB-LABEL: byte_bit_7:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_7:
; NOZBB-NOT: orc.b
; NOZBB: ret
  ; Keep the mask in registers so KnownBits can inspect it after legalization.
  %m = and i64 %x, 2155905152
  %l = shl i64 %m, 1
  %h = lshr i64 %m, 7
  %r = sub i64 %l, %h
  ret i64 %r
}

define i64 @no_known_bits(i64 %x) {
; ZBB-LABEL: no_known_bits:
; ZBB-NOT: orc.b
; ZBB: ret
; NOZBB-LABEL: no_known_bits:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, -1
  %l = shl i64 %m, 8
  %r = sub i64 %l, %m
  ret i64 %r
}

define i32 @narrow_i32(i32 %x) {
; ZBB-LABEL: narrow_i32:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: narrow_i32:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 16843009
  %l = shl i32 %m, 8
  %r = sub i32 %l, %m
  ret i32 %r
}

define signext i16 @narrow_i16(i16 %x) {
; ZBB-LABEL: narrow_i16:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: narrow_i16:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i16 %x, 257
  %l = shl i16 %m, 8
  %r = sub i16 %l, %m
  ret i16 %r
}

; Debug uses of the shift do not make it unprofitable to remove.
declare void @llvm.dbg.value(metadata, metadata, metadata)

define i64 @debug_shift(i64 %x) !dbg !4 {
; ZBB-LABEL: debug_shift:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: debug_shift:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i64 %x, 72340172838076673, !dbg !9
  %s = shl i64 %m, 8, !dbg !9
  call void @llvm.dbg.value(metadata i64 %s, metadata !7, metadata !DIExpression()), !dbg !9
  %r = sub i64 %s, %m, !dbg !9
  ret i64 %r, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "orc.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "debug_shift", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!8, !8}
!7 = !DILocalVariable(name: "shift", scope: !4, file: !1, line: 1, type: !8)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DILocation(line: 1, column: 1, scope: !4)

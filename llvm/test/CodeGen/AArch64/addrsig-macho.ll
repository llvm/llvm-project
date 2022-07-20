; RUN: llc -filetype=asm %s -o - -mtriple arm64-apple-ios -addrsig | FileCheck %s
; RUN: llc -filetype=obj %s -o %t -mtriple arm64-apple-ios -addrsig
; RUN: llvm-objdump --macho --section-headers %t | FileCheck %s --check-prefix=SECTIONS
; RUN: llvm-objdump --macho --reloc %t | FileCheck %s --check-prefix=RELOCS

; CHECK:			.addrsig{{$}}
; CHECK-NEXT:	.addrsig_sym _func03_takeaddr
; CHECK-NEXT:	.addrsig_sym _f1
; CHECK-NEXT:	.addrsig_sym _metadata_f2
; CHECK-NEXT:	.addrsig_sym _result
; CHECK-NEXT:	.addrsig_sym _g1
; CHECK-NEXT:	.addrsig_sym _a1
; CHECK-NEXT:	.addrsig_sym _i1

; The __debug_line section (which should be generated for the given input file)
; should appear immediately after the addrsig section.  Use it to make sure
; addrsig's section size has been properly set during section layout. This
; catches a regression where the next section would overlap addrsig's
; contents.
; SECTIONS:      __llvm_addrsig 00000008          [[#%.16x,ADDR:]]
; SECTIONS-NEXT: __debug_line   {{[[:xdigit:]]+}} [[#%.16x,8+ADDR]]

; RELOCS: Relocation information (__DATA,__llvm_addrsig) 7 entries
; RELOCS: address  pcrel length extern type    scattered symbolnum/value
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _i1
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _a1
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _g1
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _result
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _metadata_f2
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _f1
; RELOCS: 00000000 False ?( 3)  True   UNSIGND False     _func03_takeaddr

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

@result = global i32 0, align 4

; Function Attrs: minsize nofree noinline norecurse nounwind optsize ssp uwtable
define void @func01() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

; Function Attrs: minsize nofree noinline norecurse nounwind optsize ssp uwtable
define void @func02() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

; Function Attrs: minsize nofree noinline norecurse nounwind optsize ssp uwtable
define void @func03_takeaddr() #0 !dbg !9 {
entry:
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, 1
  store volatile i32 %add, ptr @result, align 4
  ret void
}

; Function Attrs: minsize nofree norecurse nounwind optsize ssp uwtable
define void @callAllFunctions() local_unnamed_addr {
entry:
  tail call void @func01()
  tail call void @func02()
  tail call void @func03_takeaddr()
  %0 = load volatile i32, ptr @result, align 4
  %add = add nsw i32 %0, ptrtoint (ptr @func03_takeaddr to i32)
  store volatile i32 %add, ptr @result, align 4
  ret void
}


define void()* @f1() {
  %f1 = bitcast void()* ()* @f1 to i8*
  %f2 = bitcast void()* ()* @f2 to i8*
  %f3 = bitcast void()* @f3 to i8*
  %g1 = bitcast i32* @g1 to i8*
  %g2 = bitcast i32* @g2 to i8*
  %g3 = bitcast i32* @g3 to i8*
  %dllimport = bitcast i32* @dllimport to i8*
  %tls = bitcast i32* @tls to i8*
  %a1 = bitcast i32* @a1 to i8*
  %a2 = bitcast i32* @a2 to i8*
  %i1 = bitcast void()* @i1 to i8*
  %i2 = bitcast void()* @i2 to i8*
  call void @llvm.dbg.value(metadata i8* bitcast (void()* @metadata_f1 to i8*), metadata !6, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.value(metadata i8* bitcast (void()* @metadata_f2 to i8*), metadata !6, metadata !DIExpression()), !dbg !8
  call void @f4(i8* bitcast (void()* @metadata_f2 to i8*))
  unreachable
}

declare void @f4(i8*) unnamed_addr

declare void @metadata_f1()
declare void @metadata_f2()

define internal void()* @f2() local_unnamed_addr {
  unreachable
}

declare void @f3() unnamed_addr

@g1 = global i32 0
@g2 = internal local_unnamed_addr global i32 0
@g3 = external unnamed_addr global i32

@unref = external global i32

@dllimport = external dllimport global i32

@tls = thread_local global i32 0

@a1 = alias i32, i32* @g1
@a2 = internal local_unnamed_addr alias i32, i32* @g2

@i1 = ifunc void(), void()* ()* @f1
@i2 = internal local_unnamed_addr ifunc void(), void()* ()* @f2

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { noinline }
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!5 = !DILocation(line: 0, scope: !4)
!6 = !DILocalVariable(scope: !7)
!7 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!8 = !DILocation(line: 0, scope: !7, inlinedAt: !5)
!9 = distinct !DISubprogram(scope: null, file: !1, line: 1, type: !10, unit: !0)
!10 = !DISubroutineType(types: !{})

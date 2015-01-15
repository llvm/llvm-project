; RUN: opt %s -O2 -S -o - | FileCheck %s
; Verify that we emit the same intrinsic at most once.
; rdar://problem/13056109
;
; CHECK: call void @llvm.dbg.value(metadata %struct.i14** %p
; CHECK-NOT: call void @llvm.dbg.value(metadata %struct.i14** %p
; CHECK-NEXT: call i32 @foo
; CHECK: ret
;
;
; typedef struct {
;   long i;
; } i14;
;
; int foo(i14**);
;
;   void init() {
;     i14* p = 0;
;     foo(&p);
;     p->i |= 4;
;     foo(&p);
;   }
;
; ModuleID = 'instcombine_intrinsics.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.i14 = type { i64 }

; Function Attrs: nounwind ssp uwtable
define void @init() #0 {
  %p = alloca %struct.i14*, align 8
  call void @llvm.dbg.declare(metadata %struct.i14** %p, metadata !11, metadata !{!"0x102"}), !dbg !18
  store %struct.i14* null, %struct.i14** %p, align 8, !dbg !18
  %1 = call i32 @foo(%struct.i14** %p), !dbg !19
  %2 = load %struct.i14** %p, align 8, !dbg !20
  %3 = getelementptr inbounds %struct.i14* %2, i32 0, i32 0, !dbg !20
  %4 = load i64* %3, align 8, !dbg !20
  %5 = or i64 %4, 4, !dbg !20
  store i64 %5, i64* %3, align 8, !dbg !20
  %6 = call i32 @foo(%struct.i14** %p), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @foo(%struct.i14**)

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [instcombine_intrinsics.c] [DW_LANG_C99]
!1 = !{!"instcombine_intrinsics.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00init\00init\00\007\000\001\000\006\000\000\007", !1, !5, !6, null, void ()* @init, null, null, !2} ; [ DW_TAG_subprogram ] [line 7] [def] [init]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [instcombine_intrinsics.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 1, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5.0 "}
!11 = !{!"0x100\00p\008\000", !4, !5, !12} ; [ DW_TAG_auto_variable ] [p] [line 8]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from i14]
!13 = !{!"0x16\00i14\003\000\000\000\000", !1, null, !14} ; [ DW_TAG_typedef ] [i14] [line 3, size 0, align 0, offset 0] [from ]
!14 = !{!"0x13\00\001\0064\0064\000\000\000", !1, null, null, !15, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 64, align 64, offset 0] [def] [from ]
!15 = !{!16}
!16 = !{!"0xd\00i\002\0064\0064\000\000", !1, !14, !17} ; [ DW_TAG_member ] [i] [line 2, size 64, align 64, offset 0] [from long int]
!17 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!18 = !MDLocation(line: 8, scope: !4)
!19 = !MDLocation(line: 9, scope: !4)
!20 = !MDLocation(line: 10, scope: !4)
!21 = !MDLocation(line: 11, scope: !4)
!22 = !MDLocation(line: 12, scope: !4)

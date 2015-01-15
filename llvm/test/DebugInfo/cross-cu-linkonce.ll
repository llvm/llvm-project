; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Built from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc
; $ cat a.cpp
; # 1 "func.h"
; inline int func(int i) {
;   return i * 2;
; }
; int (*x)(int) = &func;
; $ cat b.cpp
; # 1 "func.h"
; inline int func(int i) {
;   return i * 2;
; }
; int (*y)(int) = &func;

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "func"
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_subprogram

@x = global i32 (i32)* @_Z4funci, align 8
@y = global i32 (i32)* @_Z4funci, align 8

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !20, metadata !{!"0x102"}), !dbg !21
  %2 = load i32* %1, align 4, !dbg !22
  %3 = mul nsw i32 %2, 2, !dbg !22
  ret i32 %3, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !13}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19, !19}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !10, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/a.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"a.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func\00func\00_Z4funci\001\000\001\000\006\00256\000\001", !5, !6, !7, null, i32 (i32)* @_Z4funci, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!5 = !{!"func.h", !"/tmp/dbginfo"}
!6 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/func.h]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!11}
!11 = !{!"0x34\00x\00x\00\004\000\001", null, !6, !12, i32 (i32)** @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !14, !2, !2, !3, !15, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/b.cpp] [DW_LANG_C_plus_plus]
!14 = !{!"b.cpp", !"/tmp/dbginfo"}
!15 = !{!16}
!16 = !{!"0x34\00y\00y\00\004\000\001", null, !6, !12, i32 (i32)** @y, null} ; [ DW_TAG_variable ] [y] [line 4] [def]
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 2}
!19 = !{!"clang version 3.5.0 "}
!20 = !{!"0x101\00i\0016777217\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [i] [line 1]
!21 = !MDLocation(line: 1, scope: !4)
!22 = !MDLocation(line: 2, scope: !4)

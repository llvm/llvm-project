; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Generate from clang with the following source. Note that the definition of
; the inline function follows its use to workaround another bug that should be
; fixed soon.
; namespace ns {
; int func(int i);
; }
; extern int x;
; int main() { return ns::func(x); }
; int __attribute__((always_inline)) ns::func(int i) { return i * 2; }

; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN2ns4funcEi"
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_formal_parameter
; CHECK:   NULL
; CHECK-NOT: NULL
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "_ZN2ns4funcEi"
; CHECK-NOT: DW_TAG
; CHECK:     DW_TAG_formal_parameter
; CHECK:       DW_AT_abstract_origin {{.*}} "i"
; CHECK:     NULL
; CHECK:   NULL
; CHECK: NULL

@x = external global i32

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %i.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @x, align 4, !dbg !16
  store i32 %0, i32* %i.addr.i, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr.i, metadata !17, metadata !{!"0x102"}), !dbg !18
  %1 = load i32* %i.addr.i, align 4, !dbg !18
  %mul.i = mul nsw i32 %1, 2, !dbg !18
  ret i32 %mul.i, !dbg !16
}

; Function Attrs: alwaysinline nounwind uwtable
define i32 @_ZN2ns4funcEi(i32 %i) #1 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !17, metadata !{!"0x102"}), !dbg !19
  %0 = load i32* %i.addr, align 4, !dbg !19
  %mul = mul nsw i32 %0, 2, !dbg !19
  ret i32 %mul, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/namespace_inline_function_definition.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"namespace_inline_function_definition.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !9}
!4 = !{!"0x2e\00main\00main\00\005\000\001\000\006\00256\000\005", !1, !5, !6, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/namespace_inline_function_definition.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00func\00func\00_ZN2ns4funcEi\006\000\001\000\006\00256\000\006", !1, !10, !11, null, i32 (i32)* @_ZN2ns4funcEi, null, null, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!10 = !{!"0x39\00ns\001", !1, null} ; [ DW_TAG_namespace ] [ns] [line 1]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!8, !8}
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 2}
!15 = !{!"clang version 3.5.0 "}
!16 = !MDLocation(line: 5, scope: !4)
!17 = !{!"0x101\00i\0016777222\000", !9, !5, !8} ; [ DW_TAG_arg_variable ] [i] [line 6]
!18 = !MDLocation(line: 6, scope: !9, inlinedAt: !16)
!19 = !MDLocation(line: 6, scope: !9)

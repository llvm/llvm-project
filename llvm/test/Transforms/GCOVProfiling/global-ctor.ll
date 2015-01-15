; RUN: echo '!16 = !{!"%/T/global-ctor.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: not grep '_GLOBAL__sub_I_global-ctor' %T/global-ctor.gcno
; RUN: rm %T/global-ctor.gcno

@x = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_global-ctor.ll, i8* null }]

; Function Attrs: nounwind
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  %call = call i32 @_Z1fv(), !dbg !13
  store i32 %call, i32* @x, align 4, !dbg !13
  ret void, !dbg !13
}

declare i32 @_Z1fv() #1

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_global-ctor.ll() #0 section ".text.startup" {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  call void @__cxx_global_var_init(), !dbg !14
  ret void, !dbg !14
}

attributes #0 = { nounwind }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.gcov = !{!16}
!llvm.ident = !{!12}

!0 = !{!"0x11\004\00clang version 3.5.0 (trunk 210217)\000\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/nlewycky/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !"/home/nlewycky"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\002\001\001\000\006\00256\000\002", !5, !6, !7, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [__cxx_global_var_init]
!5 = !{!"global-ctor.ll", !"/home/nlewycky"}
!6 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [/home/nlewycky/global-ctor.ll]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!"0x2e\00\00\00_GLOBAL__sub_I_global-ctor.ll\000\001\001\000\006\0064\000\000", !1, !9, !7, null, void ()* @_GLOBAL__sub_I_global-ctor.ll, null, null, !2} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!9 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/home/nlewycky/<stdin>]
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 2}
!12 = !{!"clang version 3.5.0 (trunk 210217)"}
!13 = !MDLocation(line: 2, scope: !4)
!14 = !MDLocation(line: 0, scope: !15)
!15 = !{!"0xb\000", !5, !8} ; [ DW_TAG_lexical_block ] [/home/nlewycky/global-ctor.ll]

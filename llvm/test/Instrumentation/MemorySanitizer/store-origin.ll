; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS1 %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS2 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Check origin instrumentation of stores.
; Check that debug info for origin propagation code is set correctly.

; Function Attrs: nounwind
define void @Store(i32* nocapture %p, i32 %x) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %p, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !16
  store i32 %x, i32* %p, align 4, !dbg !17, !tbaa !18
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind sanitize_memory "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = !{!"0x11\0012\00clang version 3.5.0 (204220)\001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/build0/../2.cc] [DW_LANG_C99]
!1 = !{!"../2.cc", !"/tmp/build0"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00Store\00Store\00\001\000\001\000\006\00256\001\001", !1, !5, !6, null, void (i32*, i32)* @Store, null, null, !10} ; [ DW_TAG_subprogram ] [line 1] [def] [Store]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/build0/../2.cc]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8, !9}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!11, !12}
!11 = !{!"0x101\00p\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [p] [line 1]
!12 = !{!"0x101\00x\0033554433\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [x] [line 1]
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 1, !"Debug Info Version", i32 2}
!15 = !{!"clang version 3.5.0 (204220)"}
!16 = !MDLocation(line: 1, scope: !4)
!17 = !MDLocation(line: 2, scope: !4)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !MDLocation(line: 3, scope: !4)


; CHECK: @Store
; CHECK: load {{.*}} @__msan_param_tls
; CHECK: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK: store {{.*}}!dbg ![[DBG:[01-9]+]]
; CHECK: icmp
; CHECK: br i1
; CHECK: <label>

; Origin tracking level 1: simply store the origin value
; CHECK-ORIGINS1: store i32 {{.*}}[[ORIGIN]],{{.*}}!dbg !{{.*}}[[DBG]]

; Origin tracking level 2: pass origin value through __msan_chain_origin and store the result.
; CHECK-ORIGINS2: [[ORIGIN2:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 {{.*}}[[ORIGIN]])
; CHECK-ORIGINS2: store i32 {{.*}}[[ORIGIN2]],{{.*}}!dbg !{{.*}}[[DBG]]

; CHECK: br label{{.*}}!dbg !{{.*}}[[DBG]]
; CHECK: <label>
; CHECK: store{{.*}}!dbg !{{.*}}[[DBG]]
; CHECK: ret void

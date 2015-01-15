; REQUIRES: object-emission

; RUN: llc -mtriple=i386-linux-gnu -filetype=obj -relocation-model=pic %s -o /dev/null

; Derived from the test case in PR20367, there's nothing really positive to
; test here (hence no FileCheck, etc). All that was wrong is that the debug info
; intrinsics (introduced by inlining) in 'f1' were causing codegen to crash, but
; since 'f1' is a nodebug function, there's no positive outcome to confirm, just
; that debug info doesn't get in the way/cause a crash.

; The test case isn't particularly well reduced/tidy, but as simple as I could
; get the C++ source. I assume the complexity is mostly just about producing a
; certain amount of register pressure, so it might be able to be simplified/made
; more uniform.

; Generated from:
; $ clang-tot -cc1 -triple i386 -emit-obj -g -O3 repro.cpp
; void sink(const void *);
; int source();
; void f3(int);
; 
; extern bool b;
; 
; struct string {
;   unsigned *mem;
; };
; 
; extern string &str;
; 
; inline __attribute__((always_inline)) void s2(string *lhs) { sink(lhs->mem); }
; inline __attribute__((always_inline)) void f() {
;   string str2;
;   s2(&str2);
;   sink(&str2);
; }
; void __attribute__((nodebug)) f1() {
;   for (int iter = 0; iter != 2; ++iter) {
;     f();
;     sink(str.mem);
;     if (b) return;
;   }
; }

%struct.string = type { i32* }

@str = external constant %struct.string*
@b = external global i8

; Function Attrs: nounwind
define void @_Z2f1v() #0 {
entry:
  %str2.i = alloca %struct.string, align 4
  %0 = bitcast %struct.string* %str2.i to i8*, !dbg !26
  %1 = load %struct.string** @str, align 4
  %mem = getelementptr inbounds %struct.string* %1, i32 0, i32 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iter.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @llvm.lifetime.start(i64 4, i8* %0), !dbg !26
  call void @llvm.dbg.value(metadata %struct.string* %str2.i, i64 0, metadata !16, metadata !{!"0x102"}) #3, !dbg !26
  call void @llvm.dbg.value(metadata %struct.string* %str2.i, i64 0, metadata !27, metadata !{!"0x102"}) #3, !dbg !29
  call void @_Z4sinkPKv(i8* undef) #3, !dbg !29
  call void @_Z4sinkPKv(i8* %0) #3, !dbg !30
  call void @llvm.lifetime.end(i64 4, i8* %0), !dbg !31
  %2 = load i32** %mem, align 4, !tbaa !32
  %3 = bitcast i32* %2 to i8*
  call void @_Z4sinkPKv(i8* %3) #3
  %4 = load i8* @b, align 1, !tbaa !37, !range !39
  %tobool = icmp ne i8 %4, 0
  %inc = add nsw i32 %iter.02, 1
  %cmp = icmp eq i32 %inc, 2
  %or.cond = or i1 %tobool, %cmp
  br i1 %or.cond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @_Z4sinkPKv(i8*) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = !{!"0x11\004\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !3, !10, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00string\007\0032\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS6string"} ; [ DW_TAG_structure_type ] [string] [line 7, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"repro.cpp", !"/tmp/dbginfo"}
!6 = !{!7}
!7 = !{!"0xd\00mem\008\0032\0032\000\000", !5, !"_ZTS6string", !8} ; [ DW_TAG_member ] [mem] [line 8, size 32, align 32, offset 0] [from ]
!8 = !{!"0xf\00\000\0032\0032\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from unsigned int]
!9 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!10 = !{!11, !17}
!11 = !{!"0x2e\00f\00f\00_Z1fv\0014\000\001\000\006\00256\001\0014", !5, !12, !13, null, null, null, null, !15} ; [ DW_TAG_subprogram ] [line 14] [def] [f]
!12 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/repro.cpp]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{null}
!15 = !{!16}
!16 = !{!"0x100\00str2\0015\000", !11, !12, !"_ZTS6string"} ; [ DW_TAG_auto_variable ] [str2] [line 15]
!17 = !{!"0x2e\00s2\00s2\00_Z2s2P6string\0013\000\001\000\006\00256\001\0013", !5, !12, !18, null, null, null, null, !21} ; [ DW_TAG_subprogram ] [line 13] [def] [s2]
!18 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = !{null, !20}
!20 = !{!"0xf\00\000\0032\0032\000\000", null, null, !"_ZTS6string"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from _ZTS6string]
!21 = !{!22}
!22 = !{!"0x101\00lhs\0016777229\000", !17, !12, !20} ; [ DW_TAG_arg_variable ] [lhs] [line 13]
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 2}
!25 = !{!"clang version 3.5.0 "}
!26 = !MDLocation(line: 15, scope: !11)
!27 = !{!"0x101\00lhs\0016777229\000", !17, !12, !20, !28} ; [ DW_TAG_arg_variable ] [lhs] [line 13]
!28 = !MDLocation(line: 16, scope: !11)
!29 = !MDLocation(line: 13, scope: !17, inlinedAt: !28)
!30 = !MDLocation(line: 17, scope: !11)
!31 = !MDLocation(line: 18, scope: !11)
!32 = !{!33, !34, i64 0}
!33 = !{!"_ZTS6string", !34, i64 0}
!34 = !{!"any pointer", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !{!38, !38, i64 0}
!38 = !{!"bool", !35, i64 0}
!39 = !{i8 0, i8 2}

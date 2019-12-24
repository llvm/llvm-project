; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; Source code:
;   union sk_buff {
;     int i;
;     struct {
;       int netid;
;       union {
;         int dev_id;
;         int others;
;       } dev;
;     } u;
;   };
;   #define _(x) (__builtin_preserve_access_index(x))
;   static int (*bpf_probe_read)(void *dst, int size, void *unsafe_ptr)
;       = (void *) 4;
;
;   int bpf_prog(union sk_buff *ctx) {
;     int dev_id;
;     bpf_probe_read(&dev_id, sizeof(int), _(&ctx->u.dev.dev_id));
;     return dev_id;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%union.sk_buff = type { %struct.anon }
%struct.anon = type { i32, %union.anon }
%union.anon = type { i32 }

; Function Attrs: nounwind
define dso_local i32 @bpf_prog(%union.sk_buff*) local_unnamed_addr #0 !dbg !15 {
  %2 = alloca i32, align 4
  call void @llvm.dbg.value(metadata %union.sk_buff* %0, metadata !32, metadata !DIExpression()), !dbg !34
  %3 = bitcast i32* %2 to i8*, !dbg !35
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3) #4, !dbg !35
  %4 = tail call %union.sk_buff* @llvm.preserve.union.access.index.p0s_union.sk_buffs.p0s_union.sk_buffs(%union.sk_buff* %0, i32 1), !dbg !36, !llvm.preserve.access.index !19
  %5 = getelementptr inbounds %union.sk_buff, %union.sk_buff* %4, i64 0, i32 0, !dbg !36
  %6 = tail call %union.anon* @llvm.preserve.struct.access.index.p0s_union.anons.p0s_struct.anons(%struct.anon* %5, i32 1, i32 1), !dbg !36, !llvm.preserve.access.index !23
  %7 = tail call %union.anon* @llvm.preserve.union.access.index.p0s_union.anons.p0s_union.anons(%union.anon* %6, i32 0), !dbg !36, !llvm.preserve.access.index !27
  %8 = bitcast %union.anon* %7 to i8*, !dbg !36
  %9 = call i32 inttoptr (i64 4 to i32 (i8*, i32, i8*)*)(i8* nonnull %3, i32 4, i8* %8) #4, !dbg !37
  %10 = load i32, i32* %2, align 4, !dbg !38, !tbaa !39
  call void @llvm.dbg.value(metadata i32 %10, metadata !33, metadata !DIExpression()), !dbg !34
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3) #4, !dbg !43
  ret i32 %10, !dbg !44
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   168
; CHECK-NEXT:        .long   168
; CHECK-NEXT:        .long   105
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_UNION(id = 2)
; CHECK-NEXT:        .long   83886082                # 0x5000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   13                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   0                       # BTF_KIND_STRUCT(id = 4)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   17
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   0                       # BTF_KIND_UNION(id = 5)
; CHECK-NEXT:        .long   83886082                # 0x5000002
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   27
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   34
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 6)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   41
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   45                      # BTF_KIND_FUNC(id = 7)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "sk_buff"               # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   105                     # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   117                     # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=13
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "netid"                 # string offset=17
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "dev"                   # string offset=23
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "dev_id"                # string offset=27
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "others"                # string offset=34
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "ctx"                   # string offset=41
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "bpf_prog"              # string offset=45
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=54
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/llvm/test.c" # string offset=60
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "0:1:1:0"               # string offset=97
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   8                       # FuncInfo

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   54                      # Field reloc section string offset=54
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   97
; CHECK-NEXT:        .long   0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare %union.sk_buff* @llvm.preserve.union.access.index.p0s_union.sk_buffs.p0s_union.sk_buffs(%union.sk_buff*, i32 immarg) #2

; Function Attrs: nounwind readnone
declare %union.anon* @llvm.preserve.struct.access.index.p0s_union.anons.p0s_struct.anons(%struct.anon*, i32 immarg, i32 immarg) #2

; Function Attrs: nounwind readnone
declare %union.anon* @llvm.preserve.union.access.index.p0s_union.anons.p0s_union.anons(%union.anon*, i32 immarg) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "bpf_probe_read", scope: !0, file: !1, line: 12, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)"}
!15 = distinct !DISubprogram(name: "bpf_prog", scope: !1, file: !1, line: 15, type: !16, scopeLine: 15, flags: DIFlagPrototyped, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !31)
!16 = !DISubroutineType(types: !17)
!17 = !{!9, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "sk_buff", file: !1, line: 1, size: 64, elements: !20)
!20 = !{!21, !22}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !19, file: !1, line: 2, baseType: !9, size: 32)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !19, file: !1, line: 9, baseType: !23, size: 64)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !19, file: !1, line: 3, size: 64, elements: !24)
!24 = !{!25, !26}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "netid", scope: !23, file: !1, line: 4, baseType: !9, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "dev", scope: !23, file: !1, line: 8, baseType: !27, size: 32, offset: 32)
!27 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !23, file: !1, line: 5, size: 32, elements: !28)
!28 = !{!29, !30}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "dev_id", scope: !27, file: !1, line: 6, baseType: !9, size: 32)
!30 = !DIDerivedType(tag: DW_TAG_member, name: "others", scope: !27, file: !1, line: 7, baseType: !9, size: 32)
!31 = !{!32, !33}
!32 = !DILocalVariable(name: "ctx", arg: 1, scope: !15, file: !1, line: 15, type: !18)
!33 = !DILocalVariable(name: "dev_id", scope: !15, file: !1, line: 16, type: !9)
!34 = !DILocation(line: 0, scope: !15)
!35 = !DILocation(line: 16, column: 3, scope: !15)
!36 = !DILocation(line: 17, column: 40, scope: !15)
!37 = !DILocation(line: 17, column: 3, scope: !15)
!38 = !DILocation(line: 18, column: 10, scope: !15)
!39 = !{!40, !40, i64 0}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C/C++ TBAA"}
!43 = !DILocation(line: 19, column: 1, scope: !15)
!44 = !DILocation(line: 18, column: 3, scope: !15)
